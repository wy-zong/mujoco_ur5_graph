# pick_box_only_env.py
"""
Pick Box Only Environment - 簡化版，只有方塊夾取功能，移除所有布料相關邏輯。
基於 PickBoxEnv 修改，用於調試純物理夾取。
"""
import os
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env

from ..arm.robot import Robot, UR5e, SO101
from ..arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from ..utils import mj

from lerobot.model.kinematics import RobotKinematics


class PickBoxOnlyEnv(Env):
    """純物理夾取環境，不使用 weld constraint，只依賴摩擦力夾取 Box"""
    _name = "pick_box_only"
    _robot_type = "UR5e"
    _height = 480
    _width = 640
    _states = [
        "px",
        "py",
        "pz",
        "gripper"
    ]
    _cameras = [
        "top",
        "hand"
    ]
    _state_dim = 4
    _action_dim = 14

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500  # 與原作者一致
        self._control_hz = 25  # 與原作者一致

        self._render_mode = render_mode

        self._latest_action = None
        self._render_cache = None

        # 使用純淨場景（無布料和 SO101）
        scene_path = Path(__file__).parent.parent / Path("assets/scenes/pick_box_only_scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # UR5e 初始化
        self._robot: Robot = UR5e()
        self._robot_q = np.zeros(self._robot.dof)
        self._ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                  "wrist_2_joint", "wrist_3_joint"]
        self._robot_T = sm.SE3()
        self._T0 = sm.SE3()
        self._ur5_rot_step = 0.5

        # SO101 初始化
        self._so101: Robot = SO101()
        self._so101_q = np.zeros(self._so101.dof)
        self._so101_joint_names = ["so101_shoulder_pan", "so101_shoulder_lift", "so101_elbow_flex", 
                                   "so101_wrist_flex", "so101_wrist_roll", "so101_gripper_joint"]
        self._so101_T = sm.SE3()
        self._so101_T0 = sm.SE3()
        self._so101_rot_step = 0.5

        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        self._obj_t = np.zeros(3)

        # SO101 目標座標
        self._so101_xy = np.array([0.18, 0.0], dtype=float)
        self._so101_yaw = 0.0
        self._so101_roll = 0.0
        self._so101_pitch_offset = 0.0

        self._so101_q = np.zeros(6)
        so101_urdf_path = str(Path(__file__).parent.parent / "assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")

        self.lerobot_kinematics = RobotKinematics(
            urdf_path=so101_urdf_path,
            target_frame_name="gripper_frame_link"
        )

        print(f"[PickBoxOnlyEnv] 初始化完成，純物理夾取模式")

    def reset(self):
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # UR5e 重置
        self._robot.disable_base()
        self._robot.disable_tool()

        self._robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "ur5e_base"))
        self._robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._robot.set_joint(self._robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._robot_q[i]) 
         for i, jn in enumerate(self._ur5e_joint_names)]
        
        mujoco.mj_forward(self._mj_model, self._mj_data)
        mj.attach(self._mj_model, self._mj_data, "attach", "2f85", self._robot.fkine(self._robot_q), 
                  eq_solimp=np.array([0.995, 0.995, 0.0001, 0.5, 2.0]), 
                  eq_solref=np.array([0.0003, 2.0]))
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.15))
        self._robot_T = self._robot.fkine(self._robot_q)
        self._T0 = self._robot_T.copy()

        # SO101 重置（若場景中存在）
        try:
            self._so101.disable_base()
            self._so101.disable_tool()
            try:
                self._so101.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "so101_base"))
            except:
                pass

            self._so101_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self._so101.set_joint(self._so101_q)

            for i, jn in enumerate(self._so101_joint_names):
                try:
                    mj.set_joint_q(self._mj_model, self._mj_data, jn, self._so101_q[i])
                except:
                    pass
                mujoco.mj_forward(self._mj_model, self._mj_data)

            self._so101_T = self._so101.fkine(self._so101_q)
            self._T0_so101 = self._so101_T.copy()

            # 初始化 SO101 目標
            current_q_rad = np.array([
                mj.get_joint_q(self._mj_model, self._mj_data, jn)[0] 
                for jn in self._so101_joint_names
            ])
            T_init = self.lerobot_kinematics.forward_kinematics(np.rad2deg(current_q_rad))
            T_init_se3 = sm.SE3(T_init)
            
            self._so101_target_pos = T_init_se3.t
            self._so101_target_rpy = T_init_se3.rpy(order='xyz')
        except Exception:
            # SO101 不存在，使用預設值
            self._so101_target_pos = np.array([0.18, 0.0, 0.15])
            self._so101_target_rpy = np.array([0.0, 0.0, 0.0])

        mujoco.mj_forward(self._mj_model, self._mj_data)

        # Box 隨機位置
        px_box = np.random.uniform(low=1.4, high=1.5)
        py_box = np.random.uniform(low=0.3, high=0.9)
        pz_box = 0.77
        T_Box = sm.SE3.Trans(px_box, py_box, pz_box)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        
        # 記錄 Box 位置
        self._obj_t = mj.get_body_pose(self._mj_model, self._mj_data, "Box").t
        print(f"[PickBoxOnlyEnv] Box 位置: {self._obj_t}")

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human":
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

        self._step_num = 0
        observation = self._get_observation()
        info = {"is_success": False}

        self._so101_xy[:] = [0.18, 0.0]
        self._so101_yaw = 0.0
        self._so101_roll = 0.0
        self._so101_pitch_offset = 0.0

        return observation, info

    def step(self, action):
        _t_start = time.time()
        
        n_steps = self._sim_hz // self._control_hz
        if action is not None:
            self._latest_action = action
            
            # ========== UR5 控制 ==========
            pos_off = action[0:3]
            grip_cmd = np.clip(action[3], 0.0, 1.0)

            if action.shape[0] >= 7:
                rot_raw = action[4:7]
            else:
                rot_raw = np.zeros(3)

            rot_off = np.clip(rot_raw, -1.0, 1.0) * self._ur5_rot_step

            Ti = (
                self._T0
                * sm.SE3.Trans(pos_off[0], pos_off[1], pos_off[2])
                * sm.SE3.RPY(rot_off[0], rot_off[1], rot_off[2], order="xyz")
            )

            self._robot.move_cartesian(Ti)
            joint_position = self._robot.get_joint()
            self._mj_data.ctrl[:6] = joint_position
            self._mj_data.ctrl[6] = grip_cmd * 255.0

            # ========== SO101 控制（若場景中存在）==========
            try:
                current_q_rad = np.array([
                    mj.get_joint_q(self._mj_model, self._mj_data, jn)[0] 
                    for jn in self._so101_joint_names
                ])
                current_q_deg = np.rad2deg(current_q_rad)

                dr_cmd     = float(action[7]) if action.shape[0] > 7 else 0.0
                dtheta_cmd = float(action[8]) if action.shape[0] > 8 else 0.0
                dz = float(action[9]) if action.shape[0] > 9 else 0.0
                droll_cmd  = float(action[10]) if action.shape[0] > 10 else 0.0
                dpitch_cmd = float(action[11]) if action.shape[0] > 11 else 0.0
                so101_grip_cmd = float(np.clip(action[12], 0.0, 1.0)) if action.shape[0] > 12 else 0.0

                radius_step = 0.02
                theta_step  = 0.05
                ang_step = 0.1

                curr_target_x = self._so101_target_pos[0]
                curr_target_y = self._so101_target_pos[1]
                curr_radius = np.sqrt(curr_target_x**2 + curr_target_y**2)
                curr_theta  = np.arctan2(curr_target_y, curr_target_x)

                MAX_RADIUS = 0.40 
                MIN_RADIUS = 0.10

                new_radius = curr_radius + np.clip(dr_cmd, -1.0, 1.0) * radius_step
                new_radius = np.clip(new_radius, MIN_RADIUS, MAX_RADIUS)
                new_theta  = curr_theta  + np.clip(dtheta_cmd, -1.0, 1.0) * theta_step
                
                new_target_x = new_radius * np.cos(new_theta)
                new_target_y = new_radius * np.sin(new_theta)
                
                new_target_z = self._so101_target_pos[2] + np.clip(dz, -1.0, 1.0) * 0.02
                new_target_z = np.clip(new_target_z, 0.0, 0.3)
                self._so101_target_pos = np.array([new_target_x, new_target_y, new_target_z])

                self._so101_target_rpy[2] += np.clip(droll_cmd,  -1.0, 1.0) * ang_step
                self._so101_target_rpy[1] += np.clip(dpitch_cmd, -1.0, 1.0) * ang_step
                self._so101_target_rpy[0] = new_theta

                T_target_rot = sm.SE3.RPY(self._so101_target_rpy, order='xyz')
                T_target_trans = sm.SE3.Trans(self._so101_target_pos)
                desired_ee_pose = (T_target_trans * T_target_rot).A

                ik_success = False
                try:
                    sol_q_deg = self.lerobot_kinematics.inverse_kinematics(
                        current_joint_pos=current_q_deg,
                        desired_ee_pose=desired_ee_pose,
                        position_weight=1.0,
                        orientation_weight=0.8
                    )
                    sol_q_rad = np.deg2rad(sol_q_deg)
                    ik_success = True
                except Exception as e:
                    pass  # 靜默處理 IK 錯誤

                if ik_success:
                    self._so101_q = sol_q_rad
                    for i in range(5):
                        self._mj_data.ctrl[7+i] = self._so101_q[i]
                    gripper_val = -0.5 + so101_grip_cmd * 2.0 
                    self._mj_data.ctrl[12] = gripper_val
            except Exception:
                pass  # SO101 不存在於場景中，跳過

        # 執行物理模擬
        mujoco.mj_step(self._mj_model, self._mj_data, n_steps)
        _t_physics = time.time()

        # ===== 純物理夾取 Debug 輸出 =====
        # 計算夾爪與 Box 的距離（而非布料）
        pad_pos = self._mj_data.site('right_pad').xpos
        box_pos = mj.get_body_pose(self._mj_model, self._mj_data, "Box").t
        dist = np.linalg.norm(pad_pos - box_pos)
        
        # 計算夾爪開口
        gripper_gap = np.linalg.norm(
            self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos
        )
        
        if self._step_num % 50 == 0:
            print(f"[DEBUG] dist_to_box={dist:.4f}, gripper_gap={gripper_gap:.4f}, box_z={box_pos[2]:.4f}")

        observation = self._get_observation()
        _t_obs = time.time()
        
        if self._step_num % 50 == 0:
            print(f"[PERF] physics: {(_t_physics - _t_start)*1000:.0f}ms, obs: {(_t_obs - _t_physics)*1000:.0f}ms")
        
        reward = 0.0
        terminated = False
        self._step_num += 1

        truncated = False
        if self._step_num > 10000:
            truncated = True

        info = {"is_success": terminated}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._render_mode == "human":
            self._mj_viewer.sync()

    def close(self):
        if self._mj_viewer is not None:
            self._mj_viewer.close()
        if self._mj_renderer is not None:
            try:
                self._mj_renderer.close()
            except AttributeError:
                pass

    def seed(self, seed=None):
        pass

    def _get_observation(self):
        for i in range(len(self._ur5e_joint_names)):
            self._robot_q[i] = mj.get_joint_q(self._mj_model, self._mj_data, self._ur5e_joint_names[i])[0]
        self._robot_T = self._T0.inv() * self._robot.fkine(self._robot_q)
        agent_pos = np.zeros(4, dtype=np.float32)
        agent_pos[:3] = self._robot_T.t
        agent_pos[3] = np.linalg.norm(self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos)

        self._mj_renderer.update_scene(self._mj_data, 0)
        image_top = self._mj_renderer.render()
        self._mj_renderer.update_scene(self._mj_data, 1)
        image_hand = self._mj_renderer.render()

        obs = {
            'pixels': {
                'top': image_top,
                'hand': image_hand
            },
            'agent_pos': agent_pos
        }
        self._render_cache = image_top
        return obs
