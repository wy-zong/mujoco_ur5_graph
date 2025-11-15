# scripted_flow.py
# python /home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/scripts/collect_data_teleoperation.py   --env.type=pick_box   --handler.type=scripted
from typing import Type
import numpy as np
import spatialmath as sm
import mujoco  # ← 新增，用來判斷 equality 是否啟用

from imitation_learning_lerobot.envs import Env, EnvFactory
from imitation_learning_lerobot.utils import mj  # 你專案裡的 util

def _site_SE3(data, name: str):
    R = data.site(name).xmat.reshape(3, 3)
    t = data.site(name).xpos
    # 正交化確保是 SO(3)
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    return sm.SE3.Rt(sm.SO3(R_ortho, check=False), t)

def _body_SE3(data, body_name: str):
                    R = data.body(body_name).xmat.reshape(3, 3)
                    t = data.body(body_name).xpos
                    # 正交化，避免數值小漂移
                    U, _, Vt = np.linalg.svd(R)
                    R_ortho = U @ Vt
                    return sm.SE3.Rt(sm.SO3(R_ortho, check=False), t)

def scripted_pick_and_place(env_cls: Type[Env],
                            lift: float = 0.12,
                            approach: float = 0.06,
                            steps_per_segment: int = 20,
                            grip_cmd: float = 1.0):
    """
    流程：
      到錨點上方 → 下降到錨點（等到 attach 成功）→ 抬升 lift → 橫移到對角點上方 → 下降到對角點高度
      全程不鬆爪（grip_cmd 維持 1.0）
    """
    env = env_cls(render_mode="human")
    obs, _ = env.reset()

    model, data = env._mj_model, env._mj_data

    # 取錨點位姿與對角點位姿
    # T_anchor = _site_SE3(data, 'cloth_anchor')                     # 布料錨點（在 wy 上的 site）
    T_anchor = _body_SE3(data, 'cloth_1') 
    T_diag_body = mj.get_body_pose(model, data, 'pin_D_mocap')     # 對角點（固定 mocap 的 body）
    # 上下方 waypoints
    T_anchor_above = sm.SE3.Rt(T_anchor.R, T_anchor.t + np.array([0, 0, approach]))
    T_diag_above   = sm.SE3.Rt(T_diag_body.R, T_diag_body.t + np.array([0, 0, approach]))

    # 把世界座標的 SE3 轉成 env.action（action 是：相對 T0 的平移，固定 T0 姿態）
    def world_to_action(Tw: sm.SE3):
        T_rel = env._T0.inv() * Tw
        return T_rel.t.astype(np.float32)

    # 目前相對位移（obs['agent_pos'] 前三項就是 _robot_T.t）
    def current_offset():
        return env._robot_T.t.copy().astype(np.float32)

    # 檢查是否已 attach（兩種途徑：equality 啟用 或 env 內部旗標）
    def is_attached() -> bool:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right")
        by_eq = (eq_id != -1) and (data.eq_active[eq_id] == 1)
        by_flag = getattr(env, "_cloth_attached", False)
        return bool(by_eq or by_flag)

    # 平滑插補到某個世界座標（只動 XYZ，不改姿態），一路 step；
    # 若提供 stop_when，滿足條件就提前停止。
    def go_to(Tw: sm.SE3, steps=steps_per_segment, grip=grip_cmd, stop_when=None):
        target = world_to_action(Tw)
        start  = current_offset()
        for s in range(1, steps + 1):
            a = start + (target - start) * (s / steps)
            action = np.array([a[0], a[1], a[2], grip], dtype=np.float32)
            _, _, _, _, _ = env.step(action)
            env.render()
            if stop_when is not None and stop_when():
                break

    # === 正式流程 ===
    # 1) 先到錨點上方
    go_to(T_anchor_above)
    # 2) 下降到錨點，但「等到真的 attach」才往下走
    #    - 如未附著，會一直在下降過程中持續檢查；一旦附著，立即跳出，進入抬升步驟
    go_to(T_anchor, steps=max(steps_per_segment, 20), stop_when=is_attached)

    # 3)（關鍵）只有在附著成功後才抬升。若你希望「一定要附著才繼續」可加 assert。
    #    如果你希望容忍沒附著也繼續，刪掉 assert 即可。
    assert is_attached(), "預期已附著，但未偵測到 grasp_right 或 _cloth_attached。"
    T_lift = sm.SE3.Trans(T_anchor.t[0], T_anchor.t[1], T_anchor.t[2] + lift)
    go_to(T_lift)

    # 4) 橫移到對角上方
    go_to(T_diag_above)
    # 5) 下降到對角點（放下但不鬆爪）
    go_to(T_diag_body)

    env.close()

if __name__ == "__main__":
    # 直接指定你的 env type，例如 "pick_box" 或載有布料的 Env 名稱
    EnvType = EnvFactory.get_strategies("pick_box")   # ← 若你的布料在另一個 env，換成對應名字
    scripted_pick_and_place(EnvType)
