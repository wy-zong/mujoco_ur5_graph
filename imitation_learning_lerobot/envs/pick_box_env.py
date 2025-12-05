#pick_box_env.py
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


class PickBoxEnv(Env):
    _name = "pick_box"
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
    #ur5 xyzrpy + gripper 並留下第二隻手臂的控制空間
    _state_dim = 4
    _action_dim = 14
    


    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500
        self._control_hz = 25

        self._render_mode = render_mode

        self._latest_action = None
        self._render_cache = None

        scene_path = Path(__file__).parent.parent / Path("assets/scenes/pick_box_scene_copy.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot: Robot = UR5e()
        self._robot_q = np.zeros(self._robot.dof)
        self._ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                  "wrist_2_joint", "wrist_3_joint"]
        self._robot_T = sm.SE3()
        self._T0 = sm.SE3()
        self._ur5_rot_step = 0.5  # rad

        # --- [修改 2] SO101 初始化 ---
        self._so101: Robot = SO101()
        self._so101_q = np.zeros(self._so101.dof)
        self._so101_joint_names = ["so101_shoulder_pan", "so101_shoulder_lift", "so101_elbow_flex", "so101_wrist_flex", "so101_wrist_roll", "so101_gripper_joint"]
        self._so101_T = sm.SE3()
        self._so101_T0 = sm.SE3()
        self._so101_rot_step = 0.5 # rad # 定義 SO101 的移動步長 (旋轉)

        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        self._obj_t = np.zeros(3)

        self._cloth_attached = False

        self._grasp_eq_names = ["grasp_right_0","grasp_right_1","grasp_right_2","grasp_right_3"]
        self._grasp_eq_idx = 0
        self._grasp_eq = self._grasp_eq_names[self._grasp_eq_idx]

        # Debug: Check body names
        print("Available Body Names:")
        for i in range(self._mj_model.nbody):
            print(f"- {mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, i)}")

        # --- SO101 目標平面座標與姿態（XLeRobot 風格） ---
        # 在 SO101 "shoulder 平面" 上的 2D 目標位置
        self._so101_xy = np.array([0.18, 0.0], dtype=float)  # (x_plane, y_plane)
        # base yaw / wrist roll / pitch_offset 都當成「增量控制」
        self._so101_yaw   = 0.0
        self._so101_roll  = 0.0
        self._so101_pitch_offset = 0.0

        self._so101_q = np.zeros(6)
        so101_urdf_path = "/home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

        self.lerobot_kinematics = RobotKinematics(
            urdf_path=so101_urdf_path,
            target_frame_name="gripper_frame_link" # 請確認 URDF 中的名稱
        )
        # === 在 __init__ 最後加入檢查 ===
        print("\n[DEBUG] Joint Order Check:")
        print(f"1. Your MuJoCo List: {self._so101_joint_names}")
        print(f"2. LeRobot URDF List: {self.lerobot_kinematics.joint_names}")
        
        # 檢查兩者長度是否一致
        if len(self._so101_joint_names) != len(self.lerobot_kinematics.joint_names):
            print(f"[WARN] Joint count mismatch! {len(self._so101_joint_names)} vs {len(self.lerobot_kinematics.joint_names)}")
        # ================================
        

    def next_grasp_point(self):
        self._grasp_eq_idx = (self._grasp_eq_idx + 1) % len(self._grasp_eq_names)
        self._grasp_eq = self._grasp_eq_names[self._grasp_eq_idx]
        print(f"[INFO] target point → {self._grasp_eq}")

    def reset(self):

        mujoco.mj_resetData(self._mj_model, self._mj_data)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.disable_base()
        self._robot.disable_tool()

        self._robot.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "ur5e_base"))
        self._robot_q = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        self._robot.set_joint(self._robot_q)
        [mj.set_joint_q(self._mj_model, self._mj_data, jn, self._robot_q[i]) for i, jn in
         enumerate(self._ur5e_joint_names)]
        # self._robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.15)) #移到上方避免偏轉
        mujoco.mj_forward(self._mj_model, self._mj_data)
        mj.attach(self._mj_model, self._mj_data, "attach", "2f85", self._robot.fkine(self._robot_q), eq_solimp=np.array([0.995, 0.995, 0.0001, 0.5, 2.0]), eq_solref=np.array([0.0003, 2.0]))
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.15))
        self._robot_T = self._robot.fkine(self._robot_q)
        self._T0 = self._robot_T.copy()

        # --- 重置 SO101 ---
        self._so101.disable_base()
        self._so101.disable_tool()
        try:
            self._so101.set_base(mj.get_body_pose(self._mj_model, self._mj_data, "so101_base")) # 需確認 XML body name
            print("[INFO] SO101 base body found and set from XML.")
        except:
            print("[WARN] SO101 base body not found in XML, using default base pose.")
            # pass # 若找不到 body 則假設在原點或由 XML 定義

        # SO101 初始姿勢 (全 0 為伸直，可依需求調整)
        # self._so101_q = np.array([0.0, -0.6, 1.2, -0.6, 0.0, 0.0])
        self._so101_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._so101.set_joint(self._so101_q)

        for i, jn in enumerate(self._so101_joint_names):
            try:
                mj.set_joint_q(self._mj_model, self._mj_data, jn, self._so101_q[i])
            except:
                print(f"[WARN] SO101 joint '{jn}' not found in XML, skipping setting its position.")
                # pass # 忽略 XML 中找不到的 joint
            mujoco.mj_forward(self._mj_model, self._mj_data)
        #為何這裡不需要set_tool?
        self._so101_T = self._so101.fkine(self._so101_q)
        self._T0_so101 = self._so101_T.copy()


        # ... (保留原本 SO101 reset 代碼，直到 self._so101_T 計算完畢) ...
        
        # [新增] 初始化目標記憶變數
        # 使用 LeRobot FK 算出來的初始位置做為起點
        current_q_rad = np.array([
            mj.get_joint_q(self._mj_model, self._mj_data, jn)[0] 
            for jn in self._so101_joint_names
        ])
        T_init = self.lerobot_kinematics.forward_kinematics(np.rad2deg(current_q_rad))
        T_init_se3 = sm.SE3(T_init)
        
        self._so101_target_pos = T_init_se3.t
        # 取得初始 RPY (注意 spatialmath 的 rpy 順序，通常是 'zyx' 或 'xyz'，這裡統一用 rpy)
        self._so101_target_rpy = T_init_se3.rpy(order='xyz')

        mujoco.mj_forward(self._mj_model, self._mj_data)



        # --- 夾爪已 attach 完成到 UR 法蘭 ---
        # right_pad 的當下世界位姿
        # T_right_pad = mj.get_body_pose(self._mj_model, self._mj_data, "right_pad")

        # # 一行搞定：把 wy_free 對齊 right_pad，並初始化/啟用 weld "grasp_right"
        # mj.attach(self._mj_model, self._mj_data, "grasp_right", "wy_free", T_right_pad)

        # mujoco.mj_forward(self._mj_model, self._mj_data)
        # ---
        px_box = np.random.uniform(low=1.4, high=1.5)
        py_box = np.random.uniform(low=0.3, high=0.9)
        pz_box = 0.77
        T_Box = sm.SE3.Trans(px_box, py_box, pz_box)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        

        # px_container = np.random.uniform(low=1.4, high=1.5)
        # py_container = np.random.uniform(low=0.3, high=0.9)
        # pz_container = 0.77
        # while np.linalg.norm(
        #         np.array([px_box, py_box, pz_box] - np.array([px_container, py_container, pz_container]))) < 0.2:
        #     px_container = np.random.uniform(low=1.4, high=1.5)
        #     py_container = np.random.uniform(low=0.3, high=0.9)
        #     pz_container = 0.77
        # T_container = sm.SE3.Trans(px_container, py_container, pz_container)

        # container_eq_data = np.zeros(11)
        # container_eq_data[3:6] = T_container.t
        # container_eq_data[6:10] = T_container.UnitQuaternion()
        # container_eq_data[-1] = 1.0
        # mj.attach(self._mj_model, self._mj_data, "container_attach",
        #           "container_free_joint", T_container, eq_data=container_eq_data)
        # mujoco.mj_forward(self._mj_model, self._mj_data)

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human":
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)#,
                                                           #show_left_ui=False, show_right_ui=False)

        self._step_num = 0
        observation = self._get_observation()
        info = {"is_success": False}

        self._so101_xy[:] = [0.18, 0.0]
        self._so101_yaw = 0.0
        self._so101_roll = 0.0
        self._so101_pitch_offset = 0.0

        return observation, info
    
    @staticmethod
    def detach(model: mujoco.MjModel, data: mujoco.MjData, equality_name: str) -> None:
        """
        關閉一個已存在的 equality（例如 weld）。
        同時關 data.eq_active 與 model.eq_active0（因版本差異），然後 forward + 多步 step 穩定。
        """
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
        if eq_id == -1:
            raise ValueError(f"Equality constraint '{equality_name}' not found.")

        # 1) 先把兩邊旗標都關掉（誰存在就關誰）
        if hasattr(data, "eq_active"):
            data.eq_active[eq_id] = 0
        if hasattr(model, "eq_active0"):
            model.eq_active0[eq_id] = 0

        # 2) 推進求解，讓約束真正消失
        mujoco.mj_forward(model, data)
        for _ in range(5):
            mujoco.mj_step(model, data)

        # 3) Debug：確認旗標真的關了
        v_data = (hasattr(data, "eq_active") and int(data.eq_active[eq_id])) or None
        v_model = (hasattr(model, "eq_active0") and int(model.eq_active0[eq_id])) or None
        print(f"[DEBUG] detach '{equality_name}': data.eq_active={v_data}, model.eq_active0={v_model}")


    def _is_gripper_closed(self, cmd_thresh: float = 0.5, gap_thresh: float = 0.10) -> bool:
        """
        判斷夾爪是否關閉：
        - 命令：self._mj_data.ctrl[6]（0~255）需要 >= 0.5*255
        - 實際：左右夾爪 pad 的距離要 <= gap_thresh（公尺）
        """
        # 1) 根據你在 step() 裡的寫法：ctrl[6] = action[3] * 255.0
        #    所以把 0~1 的門檻轉成 0~255 來比
        cmd_val = float(self._mj_data.ctrl[6]) if self._mj_data.ctrl.shape[0] > 6 else 0.0
        cmd_ok = cmd_val >= (cmd_thresh * 255.0)

        # 2) 量測左右 pad 的實際間距
        gap = np.linalg.norm(
            self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos
        )
        gap_ok = gap <= gap_thresh

        return cmd_ok and gap_ok


    def step(self, action):
        n_steps = self._sim_hz // self._control_hz
        if action is not None:
            self._latest_action = action
            
            # ---------- UR5: 保持原樣 ----------
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

            # ================= SO101 控制 (Latched Target 版) =================
            
            # 1. 取得當前關節 (僅用於 IK 的 seed，不用於計算目標)
            current_q_rad = np.array([
                mj.get_joint_q(self._mj_model, self._mj_data, jn)[0] 
                for jn in self._so101_joint_names
            ])
            current_q_deg = np.rad2deg(current_q_rad)

            # 2. 處理 Action 輸入
            # mapping: 7:dr_cmd, 8:dtheta_cmd, 9:dz, 10:droll, 11:dpitch, 12:grip
            dr_cmd     = float(action[7])  # 半徑增量 (前後)
            dtheta_cmd = float(action[8])  # 角度增量 (旋轉)
            dz = float(action[9])
            droll_cmd  = float(action[10])
            dpitch_cmd = float(action[11])
            so101_grip_cmd = float(np.clip(action[12], 0.0, 1.0))

            radius_step = 0.02  # 每次伸長 2cm
            theta_step  = 0.05  # 每次轉動約 3度 (弧度)   
            ang_step = 0.1    

            # 3. [核心變更] 基於「記憶中的目標」進行圓柱座標計算」
            # 讀取目前記憶的目標座標 (x, y)
            curr_target_x = self._so101_target_pos[0]
            curr_target_y = self._so101_target_pos[1]
            
            # 轉成極座標 (Radius, Theta)
            curr_radius = np.sqrt(curr_target_x**2 + curr_target_y**2)
            curr_theta  = np.arctan2(curr_target_y, curr_target_x)

            # 加上增量
            new_radius = curr_radius + np.clip(dr_cmd, -1.0, 1.0) * radius_step
            new_theta  = curr_theta  + np.clip(dtheta_cmd, -1.0, 1.0) * theta_step
            
            # 轉回直角座標 (x, y)
            new_target_x = new_radius * np.cos(new_theta)
            new_target_y = new_radius * np.sin(new_theta)
            
            # 更新 Z 軸 (直接累加)
            new_target_z = self._so101_target_pos[2] + np.clip(dz, -1.0, 1.0) * 0.02

            # 將算好的新座標寫回記憶變數 (Latched)
            self._so101_target_pos = np.array([new_target_x, new_target_y, new_target_z])

            # 4. [姿態控制] 混合「自動 Yaw」與「手動 Pitch/Roll」
            # Pitch (rpy[1]) 和 Roll (rpy[0]) 使用累加控制
            self._so101_target_rpy[2] += np.clip(droll_cmd,  -1.0, 1.0) * ang_step # Roll
            self._so101_target_rpy[1] += np.clip(dpitch_cmd, -1.0, 1.0) * ang_step # Pitch

            # Yaw (rpy[2]) 強制設定為新的底座角度 (new_theta)
            # 夾爪永遠指向圓外
            self._so101_target_rpy[0] = new_theta

            # 5. 構建矩陣並傳給 LeRobot IK
            # 這裡使用 'xyz' 順序: Rz(yaw) * Ry(pitch) * Rx(roll)
            # 剛好對應：先轉到底座角度 -> 再壓手腕 -> 再轉夾爪
            T_target_rot = sm.SE3.RPY(self._so101_target_rpy, order='xyz')
            T_target_trans = sm.SE3.Trans(self._so101_target_pos)
            
            desired_ee_pose = (T_target_trans * T_target_rot).A

            # 5. 呼叫 LeRobot IK
            ik_success = False
            try:
                sol_q_deg = self.lerobot_kinematics.inverse_kinematics(
                    current_joint_pos=current_q_deg,
                    desired_ee_pose=desired_ee_pose,
                    position_weight=1.0,
                    orientation_weight=0.8 # 現在可以試著調回 0.1 或更高，因為目標穩定了
                )
                
                sol_q_rad = np.deg2rad(sol_q_deg)
                ik_success = True

            except Exception as e:
                print(f"LeRobot IK Error: {e}")

            # 6. 寫入控制 (保持不變)
            if ik_success:
                self._so101_q = sol_q_rad
                for i in range(5):
                    mj.set_joint_q(self._mj_model, self._mj_data, self._so101_joint_names[i], self._so101_q[i])
                
                gripper_val = -0.5 + so101_grip_cmd * 2.0 
                mj.set_joint_q(self._mj_model, self._mj_data, self._so101_joint_names[5], gripper_val)

                self._so101.set_joint(self._so101_q)
                mujoco.mj_forward(self._mj_model, self._mj_data)
            
        mujoco.mj_step(self._mj_model, self._mj_data, n_steps)
        
        

        # 靠近就連接
        # 3) 若尚未連接，檢查距離是否達標
        # if not self._cloth_attached:
        #     pad_pos = self._mj_data.site('right_pad').xpos
        #     cloth_pos = self._mj_data.site('cloth_anchor').xpos
        #     dist = np.linalg.norm(pad_pos - cloth_pos)

        #     # 門檻你可以依尺寸調
        #     if dist < 0.06:
        #         # (A) 方式一：對齊並啟用 weld（推薦）
        #         T_right_pad = mj.get_body_pose(self._mj_model, self._mj_data, "right_pad")
        #         # 這行會：把 wy_free 對齊到 right_pad，並把名為 grasp_right 的 weld 啟用
        #         mj.attach(self._mj_model, self._mj_data, "grasp_right", "wy_free", T_right_pad)

        #         # (B) 方式二：不對齊，只是打開 weld（若你不想瞬移 wy）
        #         # eq_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right")
        #         # self._mj_data.eq_active[eq_id] = 1

        #         mujoco.mj_forward(self._mj_model, self._mj_data)
        #         self._cloth_attached = True

        pad_site = 'right_pad'  # 或你現在實際存在的 site 名
        # cloth_site = 'cloth_anchor'
        cloth_1_body = 'cloth_1'

        pad_pos = self._mj_data.site(pad_site).xpos
        # cloth_pos = self._mj_data.site(cloth_site).xpos
        cloth_1_pos = self._mj_data.body(cloth_1_body).xpos
        # dist = np.linalg.norm(pad_pos - cloth_pos)
        dist = np.linalg.norm(pad_pos - cloth_1_pos)
        # print(f"[DEBUG] dist={dist:.4f}, pad={pad_pos}, cloth={cloth_pos}")
        print(f"[DEBUG] dist={dist:.4f}, pad={pad_pos}, cloth={cloth_1_pos}")

        


        if not self._cloth_attached:
            # pad_pos = self._mj_data.site('right_pad').xpos
            # cloth_pos = self._mj_data.site('cloth_anchor').xpos
            # cloth_1_pos = self._mj_data.body(cloth_1_body).xpos
            # # dist = np.linalg.norm(pad_pos - cloth_pos)
            # dist = np.linalg.norm(pad_pos - cloth_1_pos)


            # b1 = model.eq_obj1id[eq_id]
            # b2 = model.eq_obj2id[eq_id]
            # name1 = model.body(b1).name
            # name2 = model.body(b2).name

            # body2_pos = self._mj_data.body(name2).xpos
            # dist = np.linalg.norm(pad_pos - body2_pos)
            # print(f"[DEBUG] dist={dist:.4f}, pad={pad_pos}, cloth={body2_pos}  target={self._grasp_eq}")


             # 先拿 model, data 與當前目標 equality 的 body2 名稱
            model, data = self._mj_model, self._mj_data
            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, self._grasp_eq)
            assert eq_id != -1, f"equality '{self._grasp_eq}' not found"
            b1 = model.eq_obj1id[eq_id]
            b2 = model.eq_obj2id[eq_id]
            name1 = model.body(b1).name   # 例如 'pinch'（你的 body1）
            name2 = model.body(b2).name   # 例如 'cloth_0' / 'cloth_19' / ...

            # 用 right_pad 與「當前目標點的 body2」算距離
            pad_pos   = data.site('right_pad').xpos
            body2_pos = data.body(name2).xpos
            dist = np.linalg.norm(pad_pos - body2_pos)
            print(f"[DEBUG] dist={dist:.4f}, pad={pad_pos}, cloth={body2_pos}  target={self._grasp_eq}")

            if dist < 0.07 and self._is_gripper_closed():

                model, data = self._mj_model, self._mj_data

                # [新增] 用 equality 真實綁定的兩個 body 來決定相對位姿方向
                # eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right")
                eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, self._grasp_eq)
                assert eq_id != -1, "equality 'grasp_right' not found"
                # b1 = model.eq_obj1id[eq_id]
                # b2 = model.eq_obj2id[eq_id]
                # name1 = model.body(b1).name
                # name2 = model.body(b2).name
                # print("grasp_right pair =", name1, name2)  # 應印出: right_pad cloth_1

                # wy_body = mj.get_body_pose(model, data, "wy")
                cloth_1_body = mj.get_body_pose(model, data, "cloth_1")
                # print("[DEBUG] wy_body_pos:", wy_body.t)
                # print("[DEBUG] cloth_anchor_site:", data.site('cloth_anchor').xpos)
                print("[DEBUG] cloth_1_body_pos:", cloth_1_body.t)
                # print("[DEBUG] cloth_1_site:", data.body('cloth_1_body').xpos)

                # T1, T2：equality 兩邊 body 的世界位姿（仍然從 eq_obj1id/eq_obj2id 讀）
                T1 = mj.get_body_pose(model, data, name1)  # should be 'right_pad'
                T2 = mj.get_body_pose(model, data, name2)  # should be 'wy'
                def _site_SE3(data, name: str):
                    # 先拿 R,t
                    R = data.site(name).xmat.reshape(3, 3)
                    t = data.site(name).xpos
                    # 對 R 做一次正交化，避免數值漂移造成 spatialmath 拒收
                    U, _, Vt = np.linalg.svd(R)
                    R_ortho = U @ Vt  # 保證在 SO(3)
                    # 用 check=False 更保險
                    return sm.SE3.Rt(sm.SO3(R_ortho, check=False), t)

                def _body_SE3(data, body_name: str):
                    R = data.body(body_name).xmat.reshape(3, 3)
                    t = data.body(body_name).xpos
                    # 正交化，避免數值小漂移
                    U, _, Vt = np.linalg.svd(R)
                    R_ortho = U @ Vt
                    return sm.SE3.Rt(sm.SO3(R_ortho, check=False), t)
                

                # site 世界位姿（site 沒有 xquat，要用 xmat）
                # S1 = sm.SE3.Rt(data.site('right_pad').xmat.reshape(3,3), data.site('right_pad').xpos)
                # S2 = sm.SE3.Rt(data.site('cloth_anchor').xmat.reshape(3,3),  data.site('cloth_anchor').xpos)
                mujoco.mj_forward(model, data)  # 先確保最新狀態
  
                S1 = _body_SE3(data, 'right_pad')
                # S1 = _body_SE3(data, self._grasp_eq)
   
                S2 = _body_SE3(data, 'cloth_1')
                # site 在各自 body 座標的偏移
                Delta1 = T1.inv() * S1      # right_pad_site expressed in right_pad frame
                C = T2.inv() * (T1 * Delta1)       
                Delta2 = C
                # Delta2 = T2.inv() * S2      # cloth_anchor   expressed in wy frame

                

                # 讓 site 對 site 重合所需的「body 相對位姿」
                # T_rel = Delta1 * Delta2.inv()
                d = -0.017
                T_rel = (Delta1 * sm.SE3.Tz(d)) * C.inv() 
                # T_rel = Delta1 
                q_rel = T_rel.UnitQuaternion()

                # eq_data = np.zeros(11, dtype=float)
                # eq_data[0:3] = T_rel.t
                # eq_data[3:7] = np.array([q_rel.s, *q_rel.v])  # w, x, y, z
                # eq_data[-1]  = 1.0  # 你 util 的旗標，照舊

                eq_data = np.zeros(11, dtype=float)
                eq_data[3:6]  = T_rel.t
                eq_data[6:10] = np.array([q_rel.s, *q_rel.v])  # w,x,y,z
                eq_data[-1]   = 1.0


                # ----- debug: check contact point alignment -----
                P1 = (T1 * Delta1).t
                P2 = (T2 * Delta2).t
                R1 = (T1 * Delta1).R
                R2 = (T2 * Delta2).R
                pos_err = np.linalg.norm(P1 - P2)
                ang_err = np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)))
                print(f"[DEBUG] pre-activate pos_err = {pos_err:.6f} m, ang_err = {ang_err:.3f} deg")
                # -----------------------------------------------

                # 把 free joint 設成「obj2 的當下世界位姿」就好（不瞬移）
                # 由於 obj2 就是 name2，取它的當下姿態：
                T_free = T2

                # 呼叫你的 util（不改它）
                mj.attach(model, data,
                        self._grasp_eq,
                        None,   
                        None,
                        eq_data,
                        # 建議顯式傳 1D 參數（避免 2D 形狀被靜默忽略）
                        eq_solimp=np.array([0.99, 0.99, 0.001, 0.5, 1.0]),
                        eq_solref=np.array([0.0001, 1.0]))

                # [關鍵新增] 真的在「data」層啟用，並推進求解
                data.eq_active[eq_id] = 1
                mujoco.mj_forward(model, data)
                for _ in range(5):
                    mujoco.mj_step(model, data)

                self._cloth_attached = True


        # 若已經附著，但夾爪打開太多則鬆脫
        if self._cloth_attached and (not self._is_gripper_closed()):
            try:
                model, data = self._mj_model, self._mj_data  # ★補這行（之後要用 model/data）
                # 關 weld
                self.detach(self._mj_model, self._mj_data, self._grasp_eq)
                mujoco.mj_forward(self._mj_model, self._mj_data)

                # 再次檢查旗標（防止沒關成功）
                eq_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, self._grasp_eq)
                active_now = None
                if hasattr(self._mj_data, "eq_active"):
                    active_now = int(self._mj_data.eq_active[eq_id])
                if hasattr(self._mj_model, "eq_active0"):
                    active0_now = int(self._mj_model.eq_active0[eq_id])
                    active_now = active0_now if active_now is None else max(active_now, active0_now)

                if active_now and active_now != 0:
                    print("[WARN] equality still active after detach attempt.")

                # 小退一點，避免還是黏在一起（接觸/摩擦讓它看起來像沒鬆脫）
                # 這不會改你的高層策略，只是給模擬一個解耦空間
                # pad = self._mj_data.site(self._grasp_eq).xpos.copy()
                # cloth = self._mj_data.body('cloth_1').xpos.copy()
                pad = self._mj_data.site('right_pad').xpos.copy()     # 你的 pad 是 site
                b2 = model.eq_obj2id[eq_id]
                body2_name = model.body(b2).name        
                cloth = self._mj_data.body(body2_name).xpos.copy()    # 用剛剛算過的 body2_name

                away = pad - cloth
                nrm = np.linalg.norm(away)
                if nrm > 1e-9:
                    away /= nrm
                    Ti = self._T0 * sm.SE3.Trans(away[0]*0.01, away[1]*0.01, away[2]*0.01)  # 退 1 cm
                    self._robot.move_cartesian(Ti)
                    self._mj_data.ctrl[:6] = self._robot.get_joint()
                    for _ in range(5):
                        mujoco.mj_step(self._mj_model, self._mj_data)

                self._cloth_attached = False
                print("[INFO] Gripper opened → cloth detached (confirmed).")

                # ★新增：自動切到下一個點
                self.next_grasp_point()

            except Exception as e:
                print("[WARN] detach failed:", e)


        observation = self._get_observation()
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
            # self._mj_renderer.close()
            try:
                self._mj_renderer.close()
            except AttributeError:
                pass

    def seed(self, seed=None):
        pass

    def _get_observation(self):
        mujoco.mj_forward(self._mj_model, self._mj_data)

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
