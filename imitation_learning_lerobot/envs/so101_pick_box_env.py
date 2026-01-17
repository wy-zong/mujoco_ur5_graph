# so101_pick_box_env.py
"""
SO101 Pick Box Environment - Single arm manipulation with SO101 robot only.
Based on PickBoxEnv but without UR5e, simplified for SO101-only control.
"""
import os
# 設定 MuJoCo 使用 GPU 渲染 (Windows 需要 wgl 或 glfw)
os.environ["MUJOCO_GL"] = "glfw"
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env
from ..arm.robot import Robot, SO101
from ..utils import mj

from lerobot.model.kinematics import RobotKinematics


class SO101PickBoxEnv(Env):
    _name = "so101_pick_box"
    _robot_type = "SO101"
    _height = 480
    _width = 640
    _states = [
        "joint_1", "joint_2", "joint_3",
        "joint_4", "joint_5", "joint_gripper",
        "ee_x", "ee_y", "ee_z",
        "gripper_gap"
    ]
    _cameras = [
        "top",
        "hand"
    ]
    _state_dim = 10  # 6 joints + 3 ee_pos + 1 gripper_gap
    _action_dim = 7   # dr, dtheta, dz, droll, dpitch, gripper, reserved

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 600  # 600/30 = 20 步，保持高物理精度
        self._control_hz = 30  # 匹配真機 30fps 相機

        self._render_mode = render_mode

        self._latest_action = None
        self._render_cache = None

        scene_path = Path(__file__).parent.parent / Path("assets/scenes/so101_pick_box_scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # --- SO101 初始化 ---
        self._so101: Robot = SO101()
        self._so101_q = np.zeros(self._so101.dof)
        self._so101_joint_names = [
            "so101_shoulder_pan", "so101_shoulder_lift", "so101_elbow_flex",
            "so101_wrist_flex", "so101_wrist_roll", "so101_gripper_joint"
        ]
        self._so101_T = sm.SE3()
        self._so101_T0 = sm.SE3()
        self._so101_rot_step = 0.5  # rad

        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        
        # 夾爪狀態 (0.0 = 全關, 1.0 = 全開)
        self._gripper_state = 0.0  # 初始為閉合

        # 取得 SO101 base 的世界座標 (用於限制計算)
        self._so101_base_pos = np.array([1.45, 0.6, 0.745])  # 從 XML 中讀取

        # 目標點 mocap body ID (用於視覺化)
        self._target_marker_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
        )

        # Debug: Check body names
        print("[SO101PickBoxEnv] Available Body Names:")
        for i in range(self._mj_model.nbody):
            print(f"- {mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, i)}")

        # --- SO101 目標平面座標與姿態（XLeRobot 風格） ---
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

        print("\n[DEBUG] Joint Order Check:")
        print(f"1. Your MuJoCo List: {self._so101_joint_names}")
        print(f"2. LeRobot URDF List: {self.lerobot_kinematics.joint_names}")

        if len(self._so101_joint_names) != len(self.lerobot_kinematics.joint_names):
            print(f"[WARN] Joint count mismatch! {len(self._so101_joint_names)} vs {len(self.lerobot_kinematics.joint_names)}")

    def _update_target_marker(self, target_pos_local):
        """
        更新目標點視覺化位置 (紅色小球)
        target_pos_local: SO101 base 座標系下的目標位置
        """
        if self._target_marker_id == -1:
            return
        
        # 轉換到世界座標
        world_pos = self._so101_base_pos + target_pos_local
        
        # 更新 mocap body 位置
        mocap_id = self._mj_model.body_mocapid[self._target_marker_id]
        if mocap_id >= 0:
            self._mj_data.mocap_pos[mocap_id] = world_pos

    def _read_mocap_target(self):
        """
        讀取使用者拖拉的 mocap body 位置，轉換為 SO101 base 座標系下的目標。
        在 MuJoCo viewer 中，用戶可以雙擊 mocap body 然後拖拉。
        Returns: 目標位置 (相對於 SO101 base) 或 None
        """
        if self._target_marker_id == -1:
            return None
        
        mocap_id = self._mj_model.body_mocapid[self._target_marker_id]
        if mocap_id < 0:
            return None
        
        # 讀取 mocap 世界座標
        world_pos = self._mj_data.mocap_pos[mocap_id].copy()
        
        # 轉換到 SO101 base 座標系
        local_pos = world_pos - self._so101_base_pos
        
        return local_pos

    def reset(self):
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # --- 重置 SO101 ---
        self._so101.disable_base()
        self._so101.disable_tool()
        try:
            self._so101.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "so101_base"))
            print("[INFO] SO101 base body found and set from XML.")
        except:
            print("[WARN] SO101 base body not found in XML, using default base pose.")

        # SO101 初始姿勢
        self._so101_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._so101.set_joint(self._so101_q)

        for i, jn in enumerate(self._so101_joint_names):
            try:
                mj.set_joint_q(self._mj_model, self._mj_data, jn, self._so101_q[i])
            except:
                print(f"[WARN] SO101 joint '{jn}' not found in XML, skipping setting its position.")
            mujoco.mj_forward(self._mj_model, self._mj_data)

        self._so101_T = self._so101.fkine(self._so101_q)
        self._T0_so101 = self._so101_T.copy()

        # 初始化目標記憶變數
        current_q_rad = np.array([
            mj.get_joint_q(self._mj_model, self._mj_data, jn)[0]
            for jn in self._so101_joint_names
        ])
        T_init = self.lerobot_kinematics.forward_kinematics(np.rad2deg(current_q_rad))
        T_init_se3 = sm.SE3(T_init)

        self._so101_target_pos = T_init_se3.t.copy()
        self._so101_target_rpy = T_init_se3.rpy(order='xyz').copy()

        mujoco.mj_forward(self._mj_model, self._mj_data)

        # --- Box 位置由 XML 設定，不做隨機化 ---
        # 若要啟用隨機化，取消以下註釋
        # px_box = np.random.uniform(low=1.24, high=1.30)
        # py_box = np.random.uniform(low=0.55, high=0.65)
        # pz_box = 0.78
        # T_Box = sm.SE3.Trans(px_box, py_box, pz_box)
        # mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        # 讀取 Box 實際位置 (xml 設定的)
        box_body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "Box")
        box_pos = self._mj_data.xpos[box_body_id]
        print(f"[INFO] Box at: ({box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f})")

        # 初始化目標點視覺化
        self._update_target_marker(self._so101_target_pos)

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
        
        # === 檢查使用者是否拖拉了紅色目標點 ===
        # 如果 mocap 位置與記憶中的目標不同，表示使用者拖拉了它
        mocap_target = self._read_mocap_target()
        if mocap_target is not None:
            # 計算期望位置與記憶位置的差距
            diff = np.linalg.norm(mocap_target - self._so101_target_pos)
            if diff > 0.005:  # 超過 5mm 就認為使用者拖拉了
                # 使用者拖拉了紅點，更新目標位置
                self._so101_target_pos = mocap_target.copy()
                print(f"[DRAG] 使用者拖拉目標點到: {mocap_target}")
        
        if action is not None:
            self._latest_action = action

            # ================= SO101 混合控制 =================
            # 位置 (r, θ, z) → IK 解算前 3 個關節
            # Roll → 直接控制 wrist_roll (J4)
            # Pitch → 直接控制 wrist_flex (J3)
            # 這樣 roll/pitch 不會影響其他關節

            # 1. 取得當前關節
            current_q_rad = np.array([
                mj.get_joint_q(self._mj_model, self._mj_data, jn)[0]
                for jn in self._so101_joint_names
            ])

            # 2. 處理 Action 輸入 (7 維)
            # mapping: 0:dr_cmd, 1:dtheta_cmd, 2:dz, 3:droll, 4:dpitch, 5:grip, 6:reserved
            dr_cmd = float(action[0])
            dtheta_cmd = float(action[1])
            dz = float(action[2])
            droll_cmd = float(action[3])
            dpitch_cmd = float(action[4])
            so101_grip_cmd = float(np.clip(action[5], -1.0, 1.0))  # -1=關閉, +1=打開

            # === 步進速度 ===
            radius_step = 0.01   # 每次伸長 1cm
            theta_step = 0.03    # 每次轉動約 2度
            z_step = 0.01        # Z軸步進
            joint_step = 0.05    # 關節直接控制步進 (弧度)

            # 3. 位置控制：圓柱座標計算
            curr_target_x = self._so101_target_pos[0]
            curr_target_y = self._so101_target_pos[1]

            curr_radius = np.sqrt(curr_target_x**2 + curr_target_y**2)
            curr_theta = np.arctan2(curr_target_y, curr_target_x)

            MAX_RADIUS = 0.38
            MIN_RADIUS = 0.10

            new_radius = curr_radius + np.clip(dr_cmd, -1.0, 1.0) * radius_step
            new_radius = np.clip(new_radius, MIN_RADIUS, MAX_RADIUS)
            
            # Theta 限制
            MAX_THETA = np.pi / 2
            MIN_THETA = -np.pi / 2
            new_theta = curr_theta + np.clip(dtheta_cmd, -1.0, 1.0) * theta_step
            new_theta = np.clip(new_theta, MIN_THETA, MAX_THETA)

            new_target_x = new_radius * np.cos(new_theta)
            new_target_y = new_radius * np.sin(new_theta)

            # Z 軸限制
            MIN_Z = -0.05
            MAX_Z = 0.25
            new_target_z = self._so101_target_pos[2] + np.clip(dz, -1.0, 1.0) * z_step
            new_target_z = np.clip(new_target_z, MIN_Z, MAX_Z)
            self._so101_target_pos = np.array([new_target_x, new_target_y, new_target_z])

            # 4. 更新目標點視覺化 (紅色小球)
            self._update_target_marker(self._so101_target_pos)

            # 5. 用 IK 解算位置 (只約束位置，不約束姿態)
            # 只用前 4 個關節 (J0-J3) 來達到位置目標
            # J4 (wrist_roll) 之後單獨控制
            current_q_deg = np.rad2deg(current_q_rad)
            
            T_target_trans = sm.SE3.Trans(self._so101_target_pos)
            # 使用當前姿態，不改變
            desired_ee_pose = T_target_trans.A

            ik_success = False
            try:
                # orientation_weight=0 表示完全不約束姿態，只解位置
                sol_q_deg = self.lerobot_kinematics.inverse_kinematics(
                    current_joint_pos=current_q_deg,
                    desired_ee_pose=desired_ee_pose,
                    position_weight=1.0,
                    orientation_weight=0.0  # 只約束位置！
                )
                sol_q_rad = np.deg2rad(sol_q_deg)
                ik_success = True
            except Exception as e:
                print(f"LeRobot IK Error: {e}")
                sol_q_rad = current_q_rad.copy()

            # 6. 直接控制 Roll (wrist_roll, J4) 和 Pitch (wrist_flex, J3)
            # 從 MuJoCo 讀取關節限制
            # wrist_flex (J3): range="[-1.658, 1.658]"
            # wrist_roll (J4): range="[-2.744, 2.841]"
            
            # Pitch: 直接增量控制 wrist_flex (索引 3)
            PITCH_MIN = -1.5
            PITCH_MAX = 1.5
            new_wrist_flex = current_q_rad[3] + np.clip(dpitch_cmd, -1.0, 1.0) * joint_step
            new_wrist_flex = np.clip(new_wrist_flex, PITCH_MIN, PITCH_MAX)
            
            # Roll: 直接增量控制 wrist_roll (索引 4)
            ROLL_MIN = -2.5
            ROLL_MAX = 2.5
            new_wrist_roll = current_q_rad[4] + np.clip(droll_cmd, -1.0, 1.0) * joint_step
            new_wrist_roll = np.clip(new_wrist_roll, ROLL_MIN, ROLL_MAX)

            # 7. 合併控制命令
            # IK 解出的 J0-J2 用於位置控制
            # J3 (wrist_flex) 用直接控制
            # J4 (wrist_roll) 用直接控制
            if ik_success:
                # 只用 IK 解出的前 3 個關節 (shoulder_pan, shoulder_lift, elbow_flex)
                self._mj_data.ctrl[0] = sol_q_rad[0]  # shoulder_pan
                self._mj_data.ctrl[1] = sol_q_rad[1]  # shoulder_lift
                self._mj_data.ctrl[2] = sol_q_rad[2]  # elbow_flex
            
            # 直接控制手腕關節
            self._mj_data.ctrl[3] = new_wrist_flex   # wrist_flex (Pitch)
            self._mj_data.ctrl[4] = new_wrist_roll   # wrist_roll (Roll)
            
            # 夾爪：增量控制
            # so101_grip_cmd: +1 (9/PageUp) = 打開方向, -1 (3/PageDown) = 關閉方向
            # gripper_state: 0.0 = 全關, 1.0 = 全開
            gripper_step = 0.05  # 每步增量
            if so101_grip_cmd > 0.1:  # 9/PageUp = 打開
                self._gripper_state = min(1.0, self._gripper_state + gripper_step)
            elif so101_grip_cmd < -0.1:  # 3/PageDown = 關閉
                self._gripper_state = max(0.0, self._gripper_state - gripper_step)
            # 如果 so101_grip_cmd == 0，保持當前狀態
        
        # 夾爪控制在 action 區塊外執行，確保每步都更新
        # MuJoCo 控制值：較大值 = 張開，較小值 = 閉合
        # gripper_state: 0.0 = 全關 -> ctrl=-0.5, 1.0 = 全開 -> ctrl=1.5
        gripper_val = -0.5 + self._gripper_state * 2.0
        self._mj_data.ctrl[5] = gripper_val

        mujoco.mj_step(self._mj_model, self._mj_data, n_steps)
        _t_physics = time.time()

        observation = self._get_observation()
        _t_obs = time.time()

        # 每 50 步打印一次性能資訊
        if self._step_num % 50 == 0:
            print(f"[PERF] physics: {(_t_physics - _t_start)*1000:.0f}ms, obs: {(_t_obs - _t_physics)*1000:.0f}ms, total: {(_t_obs - _t_start)*1000:.0f}ms")

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
        # 讀取關節角度
        joint_angles = np.array([
            mj.get_joint_q(self._mj_model, self._mj_data, jn)[0]
            for jn in self._so101_joint_names
        ], dtype=np.float32)

        # 計算末端位置
        current_q_deg = np.rad2deg(joint_angles)
        try:
            T_ee = self.lerobot_kinematics.forward_kinematics(current_q_deg)
            ee_pos = T_ee[:3, 3].astype(np.float32)
        except:
            ee_pos = np.zeros(3, dtype=np.float32)

        # 夾爪間距 (使用 gripper joint 的值近似)
        gripper_gap = float(joint_angles[5])  # gripper joint value

        # agent_pos: 6 joints + 3 ee_pos + 1 gripper_gap = 10 維
        agent_pos = np.concatenate([joint_angles, ee_pos, [gripper_gap]]).astype(np.float32)

        # 渲染相機畫面
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
