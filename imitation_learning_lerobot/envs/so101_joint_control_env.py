# so101_joint_control_env.py
"""
SO101 Joint Control Environment - 關節角度控制版本
專為資料蒐集設計，與 collect_data.py 架構相容。

控制模式：
- step(action): 直接關節角度控制（6 維角度，單位：度）
- run(): 執行腳本化夾取任務，返回 observations 和 actions 資料
"""
import os
os.environ["MUJOCO_GL"] = "glfw"
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env
from ..utils import mj
from lerobot.model.kinematics import RobotKinematics


class SO101JointControlEnv(Env):
    """
    SO101 Joint Control Environment
    
    與 collect_data.py 架構相容，實現 run() 方法進行資料蒐集。
    - 輸入 action: 6 維關節角度目標（度）
    - 輸出 observation: 匹配模型訓練格式
    """
    _name = "so101_pick_cube"
    _robot_type = "SO101"
    _height = 480
    _width = 640
    _states = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper"
    ]
    _cameras = [
        "top",   # 後方高處俯視相機
        "hand"   # 跟隨夾爪相機
    ]
    _state_dim = 6   # 6 joints
    _action_dim = 6  # 6 joint targets (degrees)
    _control_hz = 30  # 匹配真機控制頻率

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 600   # 物理模擬頻率
        self._control_hz = 30  # 控制頻率（匹配真機）

        self._render_mode = render_mode
        self._render_cache = None
        self._initialized = False

        # 載入場景
        scene_path = Path(__file__).parent.parent / Path("assets/scenes/so101_pick_box_scene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # SO101 關節名稱
        self._so101_joint_names = [
            "so101_shoulder_pan", "so101_shoulder_lift", "so101_elbow_flex",
            "so101_wrist_flex", "so101_wrist_roll", "so101_gripper_joint"
        ]

        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        
        # SO101 基座位置（世界座標）
        self._so101_base_pos = np.array([1.45, 0.6, 0.745])
        
        # 方塊位置（會在 soft_reset 中更新）
        self._box_positions = [
            np.array([1.70, 0.6, 0.78]),    # 預設位置
            np.array([1.72, 0.55, 0.78]),   # 右前
            np.array([1.72, 0.65, 0.78]),   # 左前
            np.array([1.68, 0.52, 0.78]),   # 右側
            np.array([1.68, 0.68, 0.78]),   # 左側
        ]
        self._current_box_idx = 0
        
        # 初始化逆運動學
        so101_urdf_path = str(Path(__file__).parent.parent / "assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
        self.lerobot_kinematics = RobotKinematics(
            urdf_path=so101_urdf_path,
            target_frame_name="gripper_frame_link"
        )
        
        print(f"[SO101JointControlEnv] Initialized with IK support")
        print(f"[SO101JointControlEnv] IK joint_names: {self.lerobot_kinematics.joint_names}")

    def reset(self):
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # 初始化關節位置為零
        initial_q = np.zeros(6)
        for i, jn in enumerate(self._so101_joint_names):
            try:
                mj.set_joint_q(self._mj_model, self._mj_data, jn, initial_q[i])
            except:
                print(f"[WARN] Joint '{jn}' not found, skipping.")
        
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # 初始化渲染器
        if self._mj_renderer is None:
            self._mj_renderer = mujoco.renderer.Renderer(
                self._mj_model, height=self._height, width=self._width
            )
        if self._render_mode == "human" and self._mj_viewer is None:
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

        self._step_num = 0
        observation = self._get_observation()
        info = {"is_success": False}

        return observation, info

    def _soft_reset_objects_and_time(self):
        """軟重置：重新擺放 Box，清時間與步數，但保持渲染器"""
        # 重置時間
        self._mj_data.time = 0.0
        self._step_num = 0
        
        # 重置關節到初始位置
        for jn in self._so101_joint_names:
            try:
                mj.set_joint_q(self._mj_model, self._mj_data, jn, 0.0)
            except:
                pass
        
        # 設定 Box 位置
        box_pos = self._box_positions[self._current_box_idx % len(self._box_positions)]
        box_jnt_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "Box")
        if box_jnt_id >= 0:
            qpos_addr = self._mj_model.jnt_qposadr[box_jnt_id]
            self._mj_data.qpos[qpos_addr:qpos_addr+3] = box_pos
            self._mj_data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        
        self._current_box_idx += 1
        
        mujoco.mj_forward(self._mj_model, self._mj_data)
        
        observation = self._get_observation()
        info = {"is_success": False}
        return observation, info

    def _get_box_pos_local(self):
        """
        取得方塊相對於 FK 座標系的位置
        
        重要：這個座標系與 SO101 基座的世界座標不同，
        是 placo IK solver 使用的 URDF 虛擬座標系。
        """
        box_body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "Box")
        if box_body_id >= 0:
            box_world_pos = self._mj_data.xpos[box_body_id].copy()
            # 方塊相對於 SO101 基座的位置
            box_local_pos = box_world_pos - self._so101_base_pos
            return box_local_pos
        return np.array([0.25, 0.0, 0.05])  # 預設

    def _compute_ik(self, target_pos_local, current_q_deg, orientation_weight=0.0):
        """
        使用逆運動學計算達到目標位置的關節角度
        
        Args:
            target_pos_local: 相對於 SO101 基座的目標位置
            current_q_deg: 當前關節角度（度）
            orientation_weight: 姿態約束權重 (0.0 = 不約束, >0 = 約束姿態)
        
        Returns:
            關節角度（度）或 None（失敗時）
        """
        # 建立目標姿態矩陣
        if orientation_weight > 0:
            # 夾爪垂直向下：繞 Y 軸旋轉 90 度
            R_down = sm.SO3.Ry(np.pi/2)
            T_target = sm.SE3.Rt(R_down, target_pos_local)
            desired_ee_pose = T_target.A
        else:
            T_target = sm.SE3.Trans(target_pos_local)
            desired_ee_pose = T_target.A
        
        # 確保類型是 float64（placo C++ 綁定需要）
        q_deg_float64 = np.asarray(current_q_deg, dtype=np.float64)
        
        try:
            sol_q_deg = self.lerobot_kinematics.inverse_kinematics(
                current_joint_pos=q_deg_float64,
                desired_ee_pose=desired_ee_pose,
                position_weight=1.0,
                orientation_weight=orientation_weight
            )
            return sol_q_deg
        except Exception as e:
            print(f"[IK Error] {e}")
            import traceback
            traceback.print_exc()
            return None

    def step(self, action):
        """
        關節角度控制
        
        Args:
            action: 6 維向量，關節角度目標（度）
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        n_steps = self._sim_hz // self._control_hz

        if action is not None:
            # 將角度從度轉換為弧度，設定到控制器
            for i in range(min(len(action), 6)):
                angle_rad = np.deg2rad(float(action[i]))
                self._mj_data.ctrl[i] = angle_rad

        # 執行物理步進
        mujoco.mj_step(self._mj_model, self._mj_data, n_steps)

        observation = self._get_observation()

        reward = 0.0
        terminated = False
        self._step_num += 1
        truncated = self._step_num > 10000
        info = {"is_success": terminated}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self._render_mode == "human" and self._mj_viewer is not None:
            self._mj_viewer.sync()

    def close(self):
        if self._mj_viewer is not None:
            self._mj_viewer.close()
            self._mj_viewer = None
        if self._mj_renderer is not None:
            try:
                self._mj_renderer.close()
            except AttributeError:
                pass
            self._mj_renderer = None

    def seed(self, seed=None):
        pass

    def _get_observation(self):
        """
        返回與 collect_data.py 格式一致的觀測
        """
        # 讀取關節角度（弧度 -> 度）
        joint_angles_rad = np.array([
            mj.get_joint_q(self._mj_model, self._mj_data, jn)[0]
            for jn in self._so101_joint_names
        ], dtype=np.float32)
        
        joint_angles_deg = np.rad2deg(joint_angles_rad).astype(np.float32)

        # 渲染相機畫面
        top_cam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
        self._mj_renderer.update_scene(self._mj_data, top_cam_id)
        image_top = self._mj_renderer.render()
        
        hand_cam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "hand")
        self._mj_renderer.update_scene(self._mj_data, hand_cam_id)
        image_hand = self._mj_renderer.render()

        obs = {
            'pixels': {
                'top': image_top,
                'hand': image_hand
            },
            'agent_pos': joint_angles_deg
        }
        
        self._render_cache = image_top
        return obs

    def run(self, keep_state: bool = False):
        """
        執行腳本化夾取任務（使用 IK 計算軌跡）
        
        參考 UR5e 的分階段移動策略：
        1. 從當前位置開始（用 FK 獲取）
        2. 移動到方塊 XY 上方（保持高 Z）
        3. 下降到夾取高度
        4. 夾取
        5. 抬升
        6. 返回
        """
        if (not self._initialized) or (not keep_state):
            observation, info = self.reset()
            self._initialized = True
        else:
            observation, info = self._soft_reset_objects_and_time()
        
        observations = []
        actions = []
        
        # 取得初始關節角度 (使用 float64 以相容 placo)
        current_q_deg = np.rad2deg(np.array([
            mj.get_joint_q(self._mj_model, self._mj_data, jn)[0]
            for jn in self._so101_joint_names
        ], dtype=np.float64))
        
        # 用 FK 獲取當前末端位置
        T_init = self.lerobot_kinematics.forward_kinematics(current_q_deg)
        current_ee_pos = T_init[:3, 3].copy()
        print(f"[run] Initial EE position from FK: {current_ee_pos}")
        
        # 取得方塊位置（相對於 SO101 基座）
        box_local = self._get_box_pos_local()
        print(f"[run] Box local position: {box_local}")
        
        # 夾爪角度（度）
        GRIPPER_OPEN = 80.0
        GRIPPER_CLOSE = -5.0
        
        # 定義各階段的末端位置（分軸移動）
        # 階段 1: 從當前位置往上抬高（只改 Z）
        high_z = max(current_ee_pos[2], 0.18)  # 確保足夠高
        pos_high = np.array([current_ee_pos[0], current_ee_pos[1], high_z])
        
        # 階段 2: 水平移動到方塊上方（只改 XY，保持高 Z）
        pos_above_box = np.array([box_local[0]-0.01, box_local[1], high_z])
        
        # 階段 3: 下降到夾取高度（只改 Z）
        # 方塊高度約 0.035m，需要深入一點才能夾到
        grasp_z = box_local[2] - 0.03  # 方塊高度 - 偏移（深入一點）
        pos_grasp = np.array([box_local[0]-0.01, box_local[1], grasp_z])
        
        # 階段 4: 抬升（只改 Z）
        pos_lift = np.array([box_local[0], box_local[1], high_z])
        
        # 階段 5: 返回初始位置
        pos_home = current_ee_pos.copy()
        
        print(f"[run] Trajectory: {current_ee_pos} -> {pos_high} -> {pos_above_box} -> {pos_grasp} -> {pos_lift} -> {pos_home}")
        
        def move_to_target(target_pos, gripper_angle, steps=30, hold_frames=5, target_orientation_weight=0.0):
            """
            漸進式移動到目標位置
            
            Args:
                target_pos: 目標末端位置
                gripper_angle: 夾爪角度
                steps: 移動步數
                hold_frames: 到達後停留幀數
                target_orientation_weight: 目標姿態權重 (0.0 = 不約束, >0 = 漸進約束)
            """
            nonlocal observation, current_q_deg, current_ee_pos
            
            # 計算起始姿態權重（基於當前是否已約束姿態）
            # 如果目標權重 > 0，則從 0 漸進增加到目標權重
            start_ori_weight = 0.0
            
            print(f"[move] From {current_ee_pos} to {target_pos}, ori_weight: {start_ori_weight} -> {target_orientation_weight}")
            
            # 漸進式移動：位置和姿態權重都做插值
            for step in range(1, steps + 1):
                alpha = step / steps
                interp_pos = current_ee_pos + alpha * (target_pos - current_ee_pos)
                
                # 姿態權重也做插值
                interp_ori_weight = start_ori_weight + alpha * (target_orientation_weight - start_ori_weight)
                
                # 用 IK 計算
                ik_result = self._compute_ik(interp_pos, current_q_deg, orientation_weight=interp_ori_weight)
                if ik_result is not None:
                    target_q = np.zeros(6, dtype=np.float32)
                    target_q[:5] = ik_result[:5].astype(np.float32)
                    target_q[5] = gripper_angle
                    current_q_deg = target_q.copy()
                
                observations.append(observation)
                actions.append(current_q_deg.copy())
                
                observation, _, _, _, _ = self.step(current_q_deg)
                self.render()
            
            # 更新當前末端位置
            current_ee_pos = target_pos.copy()
            
            # 停留
            for _ in range(hold_frames):
                observations.append(observation)
                actions.append(current_q_deg.astype(np.float32).copy())
                observation, _, _, _, _ = self.step(current_q_deg)
                self.render()
        
        def close_gripper():
            """閉合夾爪"""
            nonlocal observation, current_q_deg
            
            target_q = current_q_deg.copy()
            target_q[5] = GRIPPER_CLOSE
            
            steps = 20
            start_q = current_q_deg.copy()
            for step in range(1, steps + 1):
                alpha = step / steps
                action = (start_q + alpha * (target_q - start_q)).astype(np.float32)
                
                observations.append(observation)
                actions.append(action.copy())
                
                observation, _, _, _, _ = self.step(action)
                self.render()
            
            current_q_deg = target_q.copy()
            
            # 停留讓夾爪穩定
            for _ in range(15):
                observations.append(observation)
                actions.append(current_q_deg.astype(np.float32).copy())
                observation, _, _, _, _ = self.step(current_q_deg)
                self.render()
        
        # 執行夾取序列
        print("[run] Phase 1: Raise up")
        move_to_target(pos_high, GRIPPER_OPEN, steps=20, hold_frames=3, target_orientation_weight=0.0)
        
        print("[run] Phase 2: Move above box and adjust orientation")
        # 移動的同時漸進調整姿態（位置和姿態同時過渡）
        move_to_target(pos_above_box, GRIPPER_OPEN, steps=60, hold_frames=5, target_orientation_weight=0.1)
        
        print("[run] Phase 3: Descend to grasp")
        move_to_target(pos_grasp, GRIPPER_OPEN, steps=50, hold_frames=5, target_orientation_weight=0.1)
        
        print("[run] Phase 4: Close gripper")
        close_gripper()
        
        print("[run] Phase 5: Lift")
        move_to_target(pos_lift, GRIPPER_CLOSE, steps=30, hold_frames=5, target_orientation_weight=0.0)
        
        print("[run] Phase 6: Return home")
        move_to_target(pos_home, GRIPPER_CLOSE, steps=40, hold_frames=10, target_orientation_weight=0.0)
        
        print(f"[run] Completed! Total frames: {len(observations)}")
        
        return {
            "observations": observations,
            "actions": actions
        }

