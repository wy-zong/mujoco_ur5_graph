# so101_pick_box_env_hybrid.py
"""
SO101 Pick Box Environment - Hybrid Control Version
混合控制版本：支援實機 Leader 手臂同步控制模擬環境，或同時控制模擬和實機

控制模式：
1. step(action): 末端點增量控制（鍵盤/搖桿/AI 推理用）
2. set_joint_positions(): 直接關節角度控制（實機同步用）
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


class SO101PickBoxEnvHybrid(Env):
    """
    SO101 Pick Box Environment - Hybrid Control Version
    
    支援兩種控制模式：
    1. step(action): 末端點增量控制（鍵盤/搖桿/AI 推理用）
    2. set_joint_positions(positions): 直接關節角度控制（實機同步用）
    """
    _name = "so101_pick_box_hybrid"
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
        print("[SO101PickBoxEnvHybrid] Available Body Names:")
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

        # =====================================================
        # 提高控制器增益以減少極端位置延遲
        # =====================================================
        self._boost_controller_gains()

    # =====================================================
    # 控制器增益調整方法
    # =====================================================
    def _boost_controller_gains(self, kp_multiplier: float = 1.0):
        """
        提高位置控制器增益以減少極端位置的延遲
        
        原始 kp = 17.8 (來自 sts3215 class)
        提高後 kp = 17.8 * 5.0 = 89.0
        
        這讓控制器在大角度變化時能更快到達目標
        """
        n_actuators = self._mj_model.nu
        print(f"\n[Controller Boost] 調整 {n_actuators} 個致動器增益...")
        
        for i in range(n_actuators):
            actuator_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            original_kp = self._mj_model.actuator_gainprm[i, 0]
            new_kp = original_kp * kp_multiplier
            self._mj_model.actuator_gainprm[i, 0] = new_kp
            
            # 同時調整 biasprm (用於位置控制器的偏置項)
            # biasprm[1] 通常等於 -kp
            if self._mj_model.actuator_biasprm[i, 1] != 0:
                self._mj_model.actuator_biasprm[i, 1] = -new_kp
            
            print(f"  [{actuator_name}] kp: {original_kp:.1f} → {new_kp:.1f}")
        
        print("[Controller Boost] 完成！\n")

    # =====================================================
    # 新增方法：直接關節控制（用於遙操作同步）
    # =====================================================
    def set_joint_positions(self, joint_positions: dict[str, float], do_physics_step: bool = True) -> dict:
        """
        直接設定關節角度（用於遙操作同步）
        
        此方法允許直接映射 follower 手臂的關節角度到模擬環境，
        而不透過末端點控制的 step() 方法。
        
        Args:
            joint_positions: 關節位置字典，格式為 {'shoulder_pan.pos': angle_deg, ...}
                           角度單位：度 (degrees)
            do_physics_step: 是否執行物理步進，預設 True
        
        Returns:
            dict: 當前觀測 (observation)
        
        Example:
            action = leader.get_action()  # {'shoulder_pan.pos': 45.0, ...}
            sim_env.set_joint_positions(action)
        """
        # LeRobot action key 到 MuJoCo control index 的映射
        joint_mapping = {
            'shoulder_pan.pos': 0,
            'shoulder_lift.pos': 1,
            'elbow_flex.pos': 2,
            'wrist_flex.pos': 3,
            'wrist_roll.pos': 4,
            'gripper.pos': 5,
        }
        
        for key, value in joint_positions.items():
            if key in joint_mapping:
                idx = joint_mapping[key]
                # 輸入是度，轉換為弧度
                angle_rad = np.deg2rad(float(value))
                self._mj_data.ctrl[idx] = angle_rad
        
        # 執行物理步進
        if do_physics_step:
            n_steps = self._sim_hz // self._control_hz
            mujoco.mj_step(self._mj_model, self._mj_data, n_steps)
        
        self._step_num += 1
        return self._get_observation()

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
        """
        末端點增量控制（原有功能，完整保留）
        
        action: 7 維向量 [dr, dtheta, dz, droll, dpitch, gripper, reserved]
        """
        _t_start = time.time()

        n_steps = self._sim_hz // self._control_hz
        
        # === 檢查使用者是否拖拉了紅色目標點 ===
        mocap_target = self._read_mocap_target()
        if mocap_target is not None:
            diff = np.linalg.norm(mocap_target - self._so101_target_pos)
            if diff > 0.005:
                self._so101_target_pos = mocap_target.copy()
                print(f"[DRAG] 使用者拖拉目標點到: {mocap_target}")
        
        if action is not None:
            self._latest_action = action

            # 1. 取得當前關節
            current_q_rad = np.array([
                mj.get_joint_q(self._mj_model, self._mj_data, jn)[0]
                for jn in self._so101_joint_names
            ])

            # 2. 處理 Action 輸入 (7 維)
            dr_cmd = float(action[0])
            dtheta_cmd = float(action[1])
            dz = float(action[2])
            droll_cmd = float(action[3])
            dpitch_cmd = float(action[4])
            so101_grip_cmd = float(np.clip(action[5], -1.0, 1.0))

            # === 步進速度 ===
            radius_step = 0.01
            theta_step = 0.03
            z_step = 0.01
            joint_step = 0.05

            # 3. 位置控制：圓柱座標計算
            curr_target_x = self._so101_target_pos[0]
            curr_target_y = self._so101_target_pos[1]

            curr_radius = np.sqrt(curr_target_x**2 + curr_target_y**2)
            curr_theta = np.arctan2(curr_target_y, curr_target_x)

            MAX_RADIUS = 0.38
            MIN_RADIUS = 0.10

            new_radius = curr_radius + np.clip(dr_cmd, -1.0, 1.0) * radius_step
            new_radius = np.clip(new_radius, MIN_RADIUS, MAX_RADIUS)
            
            MAX_THETA = np.pi / 2
            MIN_THETA = -np.pi / 2
            new_theta = curr_theta + np.clip(dtheta_cmd, -1.0, 1.0) * theta_step
            new_theta = np.clip(new_theta, MIN_THETA, MAX_THETA)

            new_target_x = new_radius * np.cos(new_theta)
            new_target_y = new_radius * np.sin(new_theta)

            MIN_Z = -0.05
            MAX_Z = 0.25
            new_target_z = self._so101_target_pos[2] + np.clip(dz, -1.0, 1.0) * z_step
            new_target_z = np.clip(new_target_z, MIN_Z, MAX_Z)
            self._so101_target_pos = np.array([new_target_x, new_target_y, new_target_z])

            # 4. 更新目標點視覺化
            self._update_target_marker(self._so101_target_pos)

            # 5. 用 IK 解算位置
            current_q_deg = np.rad2deg(current_q_rad)
            
            T_target_trans = sm.SE3.Trans(self._so101_target_pos)
            desired_ee_pose = T_target_trans.A

            ik_success = False
            try:
                sol_q_deg = self.lerobot_kinematics.inverse_kinematics(
                    current_joint_pos=current_q_deg,
                    desired_ee_pose=desired_ee_pose,
                    position_weight=1.0,
                    orientation_weight=0.0
                )
                sol_q_rad = np.deg2rad(sol_q_deg)
                ik_success = True
            except Exception as e:
                print(f"LeRobot IK Error: {e}")
                sol_q_rad = current_q_rad.copy()

            # 6. 直接控制 Roll 和 Pitch
            PITCH_MIN = -1.5
            PITCH_MAX = 1.5
            new_wrist_flex = current_q_rad[3] + np.clip(dpitch_cmd, -1.0, 1.0) * joint_step
            new_wrist_flex = np.clip(new_wrist_flex, PITCH_MIN, PITCH_MAX)
            
            ROLL_MIN = -2.5
            ROLL_MAX = 2.5
            new_wrist_roll = current_q_rad[4] + np.clip(droll_cmd, -1.0, 1.0) * joint_step
            new_wrist_roll = np.clip(new_wrist_roll, ROLL_MIN, ROLL_MAX)

            # 7. 合併控制命令
            if ik_success:
                self._mj_data.ctrl[0] = sol_q_rad[0]
                self._mj_data.ctrl[1] = sol_q_rad[1]
                self._mj_data.ctrl[2] = sol_q_rad[2]
            
            self._mj_data.ctrl[3] = new_wrist_flex
            self._mj_data.ctrl[4] = new_wrist_roll
            
            # 夾爪增量控制
            gripper_step = 0.05
            if so101_grip_cmd > 0.1:
                self._gripper_state = min(1.0, self._gripper_state + gripper_step)
            elif so101_grip_cmd < -0.1:
                self._gripper_state = max(0.0, self._gripper_state - gripper_step)
        
        # 夾爪控制
        gripper_val = -0.5 + self._gripper_state * 2.0
        self._mj_data.ctrl[5] = gripper_val

        mujoco.mj_step(self._mj_model, self._mj_data, n_steps)
        _t_physics = time.time()

        observation = self._get_observation()
        _t_obs = time.time()

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

        # 夾爪間距
        gripper_gap = float(joint_angles[5])

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
