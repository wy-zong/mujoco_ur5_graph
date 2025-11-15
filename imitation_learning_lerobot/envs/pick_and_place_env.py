import os
import time
from pathlib import Path
import numpy as np
import spatialmath as sm

import mujoco
import mujoco.viewer

from .env import Env

from ..arm.robot import Robot, UR5e
from ..arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from ..utils import mj

import imageio.v2 as imageio  # 寫 PNG 用，RGB 不會被換成 BGR

class PickAndPlaceEnv(Env):
    _name = "pick_and_place"
    _robot_type = "UR5e"
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

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()

        self._sim_hz = 500
        self._control_hz = 25

        self._render_mode = render_mode

        self._latest_action = None
        self._render_cache = None

        scene_path = Path(__file__).parent.parent / Path("assets/scenes/testscene.xml")
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(os.fspath(scene_path))
        self._mj_data: mujoco.MjData = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot: Robot = UR5e()
        self._robot_q = np.zeros(self._robot.dof)
        self._ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                  "wrist_2_joint", "wrist_3_joint"]
        self._robot_T = sm.SE3()
        self._T0 = sm.SE3()

        self._height = 480
        self._width = 640
        self._mj_renderer: mujoco.Renderer = None
        self._mj_viewer: mujoco.viewer.Handle = None

        self._step_num = 0
        self._obj_t = np.zeros(3)
         # === 放置成功判斷參數（可依需求調整） ===
        # 這裡沿用 run() 裡的目標放置點 t6 = [1.4, 0.15, 0.80]
        self._place_target_xy = np.array([1.4, 0.15], dtype=float)
        self._place_target_z  = 0.80
        self._place_tol_xy    = 0.1   # XY 半徑（3cm）
        self._place_tol_z     = 0.1   # Z 容差（±3cm）
        self._gripper_open_thresh = 0.01  # 夾爪打開閾值（依模型調）
        self._ee_box_sep_thresh  = 0.01   # 末端與盒子距離 > 5cm 視為放開

        # === 新增：觀察影像輸出路徑 ===
        self._observe_dir = Path("/home/wuc120/imitation_learning_lerobot/outputs/observe")
        self._observe_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = False  # 新增：是否已完成第一次 reset
    def _soft_reset_objects_and_time(self):
        """
        軟重置：保留上一步的機械臂/夾爪狀態，重新擺放 Box，清時間與步數。
        """

        # === 取用快照（若第一次沒有快照，就用當前值備援） ===
        # if self._last_robot_q is None:
        #     self._last_robot_q = np.array([
        #         mj.get_joint_q(self._mj_model, self._mj_data, jn)[0]
        #         for jn in self._ur5e_joint_names
        #     ], dtype=float)
        # if self._last_ctrl is None:
        #     self._last_ctrl = self._mj_data.ctrl.copy()

        # === 先 reset Data（清掉時間、暫態），但我們會立刻把手臂/夾爪狀態寫回 ===
        mujoco.mj_resetData(self._mj_model, self._mj_data)

        # === 重擺 Box（或你要 reset 的其他物件） ===
        px = np.random.uniform(low=1.25, high=1.45)
        py = np.random.uniform(low=0.3,  high=0.6)
        pz = 0.77
        T_Box = sm.SE3.Trans(px, py, pz)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)

        # === 把機械臂關節寫回 MuJoCo & 你的運動學模型 ===
        # 1) 高階運動學模型
        self._robot.set_joint(self._robot_q)
        # 2) MuJoCo qpos
        for i, jn in enumerate(self._ur5e_joint_names):
            mj.set_joint_q(self._mj_model, self._mj_data, jn, float(self._robot_q[i]))

        # # === 還原夾爪控制（以及其他 actuator ctrl，如需） ===
        # if self._last_ctrl is not None and len(self._last_ctrl) == len(self._mj_data.ctrl):
        #     
        # [:] = self._last_ctrl

        # === 更新全場景 ===
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # === 更新快取/內部紀錄 ===
        self._obj_t = mj.get_body_pose(self._mj_model, self._mj_data, "Box").t
        self._latest_action = None
        self._render_cache = None

        # 建議：不用每次重建 renderer/viewer，效能較佳；真的需要才重建
        if self._mj_renderer is None:
            try:
                self._mj_renderer.close()
            except AttributeError:
                pass
            self._mj_renderer = None
            self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)

        if self._render_mode == "human" and self._mj_viewer is None:
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

        # 清時間與步數（mj_resetData 已把 data.time 清 0；這裡清你的計數器）
        self._step_num = 0

        # 同步運動學端末端位姿（含工具偏移）
        # self._robot_T = self._robot.fkine(self._robot_q)
        # self._T0 = self._robot_T.copy()

        observation = self._get_observation()
        info = {"is_success": False}
        return observation, info

    # def _soft_reset_objects_and_time(self):

    #     mujoco.mj_resetData(self._mj_model, self._mj_data)
        
    #     mujoco.mj_forward(self._mj_model, self._mj_data)
        
        
        
    #     """保留機械臂 joint/pose，只重新擺放 Box、清時間與步數。"""
    #     # 隨機（或依你原本邏輯）擺放 Box
    #     px = np.random.uniform(low=1.25, high=1.45)
    #     py = np.random.uniform(low=0.3,  high=0.6)
    #     pz = 0.77
    #     T_Box = sm.SE3.Trans(px, py, pz)
    #     mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
    #     mujoco.mj_forward(self._mj_model, self._mj_data)
    #     self._obj_t = mj.get_body_pose(self._mj_model, self._mj_data, "Box").t
    #     self._latest_action = None
    #     self._render_cache = None

    #     if self._mj_renderer is None:   
    #         try :
    #             self._mj_renderer.close()
    #         except AttributeError:
    #             pass
    #         self._mj_renderer = None

    #     self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
    #     if self._render_mode == "human"and self._mj_viewer is None:
    #         self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

        
    #     self._step_num = 0
        
    #     observation = self._get_observation()
        
    #     info = {"is_success": False}
    #     print(f"render_mode={self._render_mode}, viewer={self._mj_viewer}")
    #     return observation, info

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
        mujoco.mj_forward(self._mj_model, self._mj_data)
        mj.attach(self._mj_model, self._mj_data, "attach", "2f85", self._robot.fkine(self._robot_q))
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.15))
        self._robot_T = self._robot.fkine(self._robot_q)
        self._T0 = self._robot_T.copy()

        # pos = np.array([[1.4, 0.5], [1.4, 0.55],[1.3, 0.45], [1.3, 0.35]],dtype=float)
        # idx = np.random.randint(len(pos))
        # px, py = pos[idx]

        # pos = np.array([[1.4, 0.5], [1.3, 0.35]], dtype=float)
        # if not hasattr(self, "_pos_i"):
        #     self._pos_i = 0
        # px, py = pos[self._pos_i % len(pos)]
        # self._pos_i += 1

        # print(px, py)
        px = np.random.uniform(low=1.25, high=1.45)
        py = np.random.uniform(low=0.3, high=0.6)

        # 固定位置
        # px =  1.45
        # py = 0.9
        # px = 1.4
        # py = 0.55
        # px =  1.45
        # py = 0.45
        pz = 0.77
        T_Box = sm.SE3.Trans(px, py, pz)
        mj.set_free_joint_pose(self._mj_model, self._mj_data, "Box", T_Box)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._obj_t = mj.get_body_pose(self._mj_model, self._mj_data, "Box").t
        self._latest_action = None
        self._render_cache = None

        if self._mj_renderer is None:   
            try :
                self._mj_renderer.close()
            except AttributeError:
                pass
            self._mj_renderer = None

        self._mj_renderer = mujoco.renderer.Renderer(self._mj_model, height=self._height, width=self._width)
        if self._render_mode == "human"and self._mj_viewer is None:
            self._mj_viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

        
        self._step_num = 0
        
        observation = self._get_observation()
        
        info = {"is_success": False}
        print(f"render_mode={self._render_mode}, viewer={self._mj_viewer}")
        return observation, info

    def step(self, action):
        n_steps = self._sim_hz // self._control_hz
        if action is not None:
            self._latest_action = action
            for i in range(n_steps):
                Ti = sm.SE3.Trans(action[0], action[1], action[2]) * sm.SE3(sm.SO3(self._T0.R))
                self._robot.move_cartesian(Ti)
                joint_position = self._robot.get_joint()
                self._mj_data.ctrl[:6] = joint_position
                action[3] = np.clip(action[3], 0, 1)
                self._mj_data.ctrl[6] = action[3] * 255.0
                mujoco.mj_step(self._mj_model, self._mj_data)

        observation = self._get_observation()
        reward = 0.0
        # terminated = False

        # self._step_num += 1

        # truncated = False
        # if self._step_num > 10000:
        #     truncated = True

        # info = {"is_success": terminated}
        # return observation, reward, terminated, truncated, info
        # === 成功/終止判斷 ===
        success = self._is_success()
        # terminated = bool(success)  # 成功就終止（若不想提前結束，可改成 False）
        terminated = False
        self._step_num += 1

        truncated = False
        if self._step_num > 10000:
            truncated = True

        info = {"is_success": success}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self._render_mode == "human":
            self._mj_viewer.sync()
            # self._mj_viewer.loop_once()

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
        self._robot_T = self._robot.fkine(self._robot_q)
        agent_pos = np.zeros(4, dtype=np.float32)
        agent_pos[:3] = self._robot_T.t
        agent_pos[3] = np.linalg.norm(self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos)

        self._mj_renderer.update_scene(self._mj_data, 0)
        image_top = self._mj_renderer.render()
        # image_top = np.moveaxis(image_top, -1, 0)
        self._mj_renderer.update_scene(self._mj_data, 1)
        image_hand = self._mj_renderer.render()
        # image_hand = np.moveaxis(image_hand, -1, 0)

        # === 新增：把影像存到 observe 資料夾 ===
        # 檔名帶 step 與時間，避免覆蓋；也方便逐步檢查
        step_str = f"{self._step_num:06d}"
        ts = time.time()
        # 確保 uint8 / RGB
        if image_top.dtype != np.uint8:
            image_top = np.clip(image_top, 0, 255).astype(np.uint8)
        if image_hand.dtype != np.uint8:
            image_hand = np.clip(image_hand, 0, 255).astype(np.uint8)
        # 寫檔
        imageio.imwrite(self._observe_dir / f"{step_str}_top_{ts:.3f}.png", image_top)
        imageio.imwrite(self._observe_dir / f"{step_str}_hand_{ts:.3f}.png", image_hand)

        obs = {
            'pixels': {
                'top': image_top,
                'hand': image_hand
            },
            'agent_pos': agent_pos
        }
        self._render_cache = image_top
        # 額外補一個 viewer.sync() 確保畫面真的刷新
        # if self._render_mode == "human" :#and self._mj_viewer is not None:
        #     self._mj_viewer.sync()
        return obs
    # 取得 Box 位置（世界座標）
    def _get_box_pos(self) -> np.ndarray:
        return mj.get_body_pose(self._mj_model, self._mj_data, "Box").t.copy()

    # 夾爪開口（left/right pad 的距離）
    def _get_gripper_opening(self) -> float:
        return float(np.linalg.norm(
            self._mj_data.site('left_pad').xpos - self._mj_data.site('right_pad').xpos
        ))

    # 末端 TCP（工具座標）位置
    def _get_ee_pos(self) -> np.ndarray:
        return self._robot_T.t.copy()

    # Box 是否在放置區域內
    def _in_place_zone(self, box_t: np.ndarray) -> bool:
        xy_ok = np.linalg.norm(box_t[:2] - self._place_target_xy) <= self._place_tol_xy
        z_ok  = (abs(box_t[2] - self._place_target_z) <= self._place_tol_z)
        return bool(xy_ok and z_ok)

    # 已釋放（夾爪打開且與 Box 分離）
    def _released(self, box_t: np.ndarray) -> bool:
        opening_ok = self._get_gripper_opening() >= self._gripper_open_thresh
        ee_sep_ok  = np.linalg.norm(self._get_ee_pos() - box_t) >= self._ee_box_sep_thresh
        return bool(opening_ok and ee_sep_ok)

    # 成功條件：Box 在放置區域內 且 已釋放
    def _is_success(self) -> bool:
        box_t = self._get_box_pos()
        return self._in_place_zone(box_t) and self._released(box_t)

    def run(self, keep_state: bool = False):


        if (not self._initialized) or (not keep_state):
            observation, info = self.reset()
            self._initialized = True
        else:
            observation, info = self._soft_reset_objects_and_time()

        # observation, info = self.reset()

        time0 = 0.04
        T0 = self._robot.get_cartesian()
        t0 = T0.t
        R0 = sm.SO3(T0.R)
        t1 = t0.copy()
        R1 = R0.copy()
        planner0 = self._cal_planner(t0, R0, t1, R1, time0)

        time1 = 2.0
        t2 = t1.copy()
        t2[:] = self._obj_t
        t2[2] = 0.86
        R2 = R1.copy()
        planner1 = self._cal_planner(t1, R1, t2, R2, time1)

        time2 = 1.0
        t3 = t2.copy()
        t3[2] = 0.78
        R3 = R2.copy()
        planner2 = self._cal_planner(t2, R2, t3, R3, time2)

        time3 = 1.0
        t4 = t3.copy()
        R4 = R3.copy()
        planner3 = self._cal_planner(t3, R3, t4, R4, time3)

        time4 = 1.0
        t5 = t4.copy()
        t5[2] = 0.86
        R5 = R4.copy()
        planner4 = self._cal_planner(t4, R4, t5, R5, time4)

        time5 = 2.0
        t6 = np.array([1.4, 0.15, 0.80])
        R6 = R5.copy()
        planner5 = self._cal_planner(t5, R5, t6, R6, time5)

        time6 = 1.0
        t7 = t6.copy()
        R7 = R6.copy()
        planner6 = self._cal_planner(t6, R6, t7, R7, time6)

        time7 = 1.0
        t8 = t7.copy()
        t8[2] = 0.90
        R8 = R7.copy()
        planner7 = self._cal_planner(t7, R7, t8, R8, time7)

        time_array = np.array([time0, time1, time2, time3, time4, time5, time6, time7])
        planner_array = [planner0, planner1, planner2, planner3, planner4, planner5, planner6, planner7]

        observations = []
        actions = []

        time_cumsum = np.cumsum(time_array)

        action = np.zeros(4, dtype=np.float32)
        planner_interpolate = np.zeros(3)
        while True:
            for j in range(len(time_cumsum)):
                if self._mj_data.time <= time_cumsum[j]:
                    if j == 0:
                        start_time = 0.0
                    else:
                        start_time = time_cumsum[j - 1]
                    planner_interpolate = planner_array[j].interpolate(self._mj_data.time - start_time).t
                    break

            else:
                self.close()
                return {
                    "observations": observations,
                    "actions": actions
                }
            action[:3] = planner_interpolate
            if self._mj_data.time >= time_cumsum[5]:
                action[3] = np.maximum(action[3] - 1.0 / time6 / self._control_hz, 0.0)
            elif self._mj_data.time >= time_cumsum[2]:
                action[3] = np.minimum(action[3] + 1.0 / time3 / self._control_hz, 1.0)

            observations.append(observation)
            actions.append(action.copy())

            observation, _, _, _, info = self.step(action)

            self.render()

    def _cal_planner(self, t0, R0, t1, R1, time):
        position_parameter = LinePositionParameter(t0, t1)
        attitude_parameter = OneAttitudeParameter(R0, R1)
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(time)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        trajectory_planner = TrajectoryPlanner(trajectory_parameter)
        return trajectory_planner


if __name__ == '__main__':
    # env = PickAndPlaceEnv(render_mode="human")
    env = PickAndPlaceEnv()
    env_data = env.run()
