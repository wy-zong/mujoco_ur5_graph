# ---- FlexFreezer: 凍結 flex 形變的小工具 ----
from dataclasses import dataclass
import numpy as np
import mujoco as mj

@dataclass
class FlexFreezeState:
    enabled: bool = False
    frozen_x: np.ndarray | None = None
    vert_slice: slice | None = None
    # 自動觸發用參數
    min_lift_z: float = 0.001      # 物體離桌面高度門檻（依需求調）
    tip_dist_thresh: float = 0.035  # 指尖距離門檻（依機構調）

class FlexFreezer:
    def __init__(self, model: mj.MjModel, flex_name: str | None = None,
                 body_name: str | None = None, left_tip_site: str | None = None,
                 right_tip_site: str | None = None):
        self.model = model
        self.body_name = body_name
        self.left_tip_site = left_tip_site
        self.right_tip_site = right_tip_site
        self.state = FlexFreezeState()

        # 若沒指定 flex_name，就拿第一個 flex（常見情況只有一個）
        if flex_name is None:
            if model.nflex > 0:
                flex_id = 0
            else:
                raise RuntimeError("模型內沒有 flexcomp，無法凍結形變。")
        else:
            flex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, flex_name)

        vstart = model.flex_vertadr[flex_id]
        vnum = model.flex_vertnum[flex_id]
        # self.state.vert_slice = slice(int(vstart*3), int((vstart+vnum)*3))
        # 新：針對 (nflexvert x 3) 的 2D 陣列，使用行 slice
        self.state.vert_slice = slice(int(vstart), int(vstart + vnum))
        

    # 兼容不同欄位名稱
    def _x_arr(self, data):
        # 先試新版/正確欄位（nflexvert x 3）
        if hasattr(data, "flexvert_xpos"):
            return data.flexvert_xpos
        # 退而求其次：有些舊測試分支可能叫 flex_x (1D 平鋪)
        if hasattr(data, "flex_x"):
            return data.flex_x
        # 最後再試極少見的 data.x
        if hasattr(data, "x"):
            return data.x
        raise AttributeError("找不到柔體頂點位置陣列：flexvert_xpos / flex_x / x 皆不存在")

    def _v_arr(self, data):
        if hasattr(data, "flexvert_xvel"):
            return data.flexvert_xvel
        if hasattr(data, "flex_v"):
            return data.flex_v
        if hasattr(data, "v"):
            return data.v
        # 沒有速度陣列也行，我們就只回寫位置
        return None


    def _tip_distance(self, data: mj.MjData) -> float:
        if not (self.left_tip_site and self.right_tip_site):
            return np.inf
        lid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.left_tip_site)
        rid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.right_tip_site)
        return float(np.linalg.norm(data.site_xpos[lid] - data.site_xpos[rid]))

    def _body_height(self, data: mj.MjData) -> float:
        if not self.body_name:
            return 0.0
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.body_name)
        return float(data.xpos[bid, 2])

    # 對外 API
    def capture_now(self, data):
        x = self._x_arr(data)
        s = self.state.vert_slice
        if x.ndim == 2:  # (nflexvert x 3)
            self.state.frozen_x = x[s, :].copy()
        else:            # 1D 平鋪
            lo, hi = s.start*3, s.stop*3
            self.state.frozen_x = x[lo:hi].copy()

        v = self._v_arr(data)
        if v is not None:
            if v.ndim == 2:
                v[s, :] = 0.0
            else:
                lo, hi = s.start*3, s.stop*3
                v[lo:hi] = 0.0
        self.state.enabled = True

    def enforce(self, data):
        if not (self.state.enabled and self.state.frozen_x is not None):
            return
        x = self._x_arr(data)
        s = self.state.vert_slice
        if x.ndim == 2:
            x[s, :] = self.state.frozen_x
        else:
            lo, hi = s.start*3, s.stop*3
            x[lo:hi] = self.state.frozen_x

        v = self._v_arr(data)
        if v is not None:
            if v.ndim == 2:
                v[s, :] = 0.0
            else:
                lo, hi = s.start*3, s.stop*3
                v[lo:hi] = 0.0


    def maybe_capture_auto(self, data: mj.MjData):
        if self.state.enabled: return
        lifted = self._body_height(data) > self.state.min_lift_z
        tips_close = self._tip_distance(data) < self.state.tip_dist_thresh
        if lifted and tips_close:
            self.capture_now(data)
