"""
SmolVLA Cross/Self-Attention 提取與靜態可視化工具
================================================
精準、無猜測的注意力提取器:
  - 從模型本身量測 vision patch grid (不用猜)
  - 精準還原每台相機的 token 區間 (扣除 image_start/end 特殊 token)
  - 分開記錄 prefix 階段與 10 步 denoise 的每一層 attention
  - 同時攔截 self-attention (VLM 內部 + expert 內部) 與 cross-attention
  - 保留 [heads, Q, K] 完整維度,下游再聚合

對應 SmolVLA 架構事實(程式碼直接驗證):
  - sample_actions 先 fill_kv_cache=True 跑一次 prefix self-attn (16 層 × 1 步)
  - 再跑 num_steps=10 次 denoise,每步每層依 self_attn_every_n_layers 選擇 self 或 cross
  - prefix token 順序:[img_start, cam1, img_end, img_start, cam2, img_end, ..., lang, state]
  - att_masks:state=True、其餘 prefix=False;padding pad_value=0
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

matplotlib.use("Agg")

try:
    import matplotlib.font_manager as fm

    for _n in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        if _n in {f.name for f in fm.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = _n
            break
except Exception:
    pass


# ============================================================
# 1. Attention Extractor
# ============================================================


class AttentionExtractor:
    """
    以 monkey-patch 的方式精準提取 SmolVLA 每個階段 / 每層 / 每種注意力。

    推理一次後,結構為::

        self.attention_maps = {
            "prefix": {
                layer_idx: {"type": "self", "probs": Tensor[heads, Q, K] fp16,
                            "q_len": int, "k_len": int},
                ...
            },
            "denoise_00": {
                layer_idx: {"type": "self"/"cross", "probs": ..., ...},
                ...
            },
            ...
        }

        self.prefix_boundaries = {
            "cameras": OrderedDict[cam_key -> (patch_start, patch_end)],
            "language": (lang_start, lang_end),     # 不含 padding
            "state":    (state_start, state_end),
            "seq_len": int,                          # 非 padded
            "padded_len": int,
        }
        self.patch_grid = (side_h, side_w)
        self.tokens_per_camera = N
        self.num_heads = int
    """

    def __init__(self, store_dtype: torch.dtype = torch.float16):
        self.attention_maps: dict[str, dict[int, dict[str, Any]]] = {}
        self.prefix_boundaries: dict[str, Any] | None = None
        self.patch_grid: tuple[int, int] | None = None
        self.tokens_per_camera: int | None = None
        self.num_heads: int | None = None
        self._store_dtype = store_dtype

        # internal patch state
        self._vlm_expert = None
        self._model_vla = None
        self._policy = None
        self._camera_keys: list[str] = []
        self._orig_eager = None
        self._orig_self = None
        self._orig_cross = None
        self._orig_embed_prefix = None
        self._orig_denoise_step = None

        # flags for eager hook to decide where to write
        self._current_phase = "prefix"
        self._current_layer = -1
        self._current_type = "self"       # "self" | "cross"
        self._denoise_step_idx = 0

    # -------- public api --------

    def register_hooks(self, model) -> None:
        try:
            vlm_expert = model.model.vlm_with_expert
            model_vla = model.model
        except AttributeError as e:
            raise RuntimeError("此模型不是 SmolVLAPolicy,缺少 model.model.vlm_with_expert") from e

        self._policy = model
        self._vlm_expert = vlm_expert
        self._model_vla = model_vla

        self._orig_eager = vlm_expert.eager_attention_forward
        self._orig_self = vlm_expert.forward_attn_layer
        self._orig_cross = vlm_expert.forward_cross_attn_layer
        self._orig_embed_prefix = model_vla.embed_prefix
        self._orig_denoise_step = model_vla.denoise_step

        # 量測 patch grid(呼叫一次 embed_image 取得 N,推導正方形邊長)
        self._measure_patch_grid()

        self._camera_keys = list(model.config.image_features)
        self.num_heads = int(vlm_expert.num_attention_heads)

        ext = self

        # ── patch 1: embed_prefix → 記錄精準邊界 ──────────────
        orig_embed_prefix = model_vla.embed_prefix

        def patched_embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
            result = orig_embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
            embs, pad_masks, att_masks = result
            ext._record_boundaries(
                images=images,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                embs=embs,
                pad_masks=pad_masks,
                att_masks=att_masks,
            )
            return result

        model_vla.embed_prefix = patched_embed_prefix

        # ── patch 2: denoise_step → 設定階段標籤 ──────────────
        orig_denoise_step = model_vla.denoise_step

        def patched_denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
            ext._current_phase = f"denoise_{ext._denoise_step_idx:02d}"
            ext.attention_maps.setdefault(ext._current_phase, {})
            try:
                return orig_denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                ext._denoise_step_idx += 1

        model_vla.denoise_step = patched_denoise_step

        # ── patch 3: forward_attn_layer → 自注意力旗標 ────────
        orig_self = vlm_expert.forward_attn_layer

        def patched_self(model_layers, inputs_embeds, layer_idx, position_ids,
                         attention_mask, batch_size, head_dim, **kw):
            ext._current_layer = int(layer_idx)
            ext._current_type = "self"
            return orig_self(model_layers, inputs_embeds, layer_idx,
                             position_ids, attention_mask, batch_size, head_dim, **kw)

        vlm_expert.forward_attn_layer = patched_self

        # ── patch 4: forward_cross_attn_layer → 交叉注意力旗標 ─
        orig_cross = vlm_expert.forward_cross_attn_layer

        def patched_cross(model_layers, inputs_embeds, layer_idx, position_ids,
                          attention_mask, batch_size, head_dim, **kw):
            ext._current_layer = int(layer_idx)
            ext._current_type = "cross"
            return orig_cross(model_layers, inputs_embeds, layer_idx,
                              position_ids, attention_mask, batch_size, head_dim, **kw)

        vlm_expert.forward_cross_attn_layer = patched_cross

        # ── patch 5: eager_attention_forward → 計算 probs 並存檔 ─
        orig_eager = vlm_expert.eager_attention_forward

        def patched_eager(attention_mask, batch_size, head_dim,
                          query_states, key_states, value_states):
            # 先呼叫原版得到 attn output (模型需要),同時我們也獨立計算 probs
            att_output = orig_eager(attention_mask, batch_size, head_dim,
                                    query_states, key_states, value_states)
            try:
                probs = ext._compute_probs(attention_mask, head_dim,
                                           query_states, key_states)
                ext._store_probs(probs)
            except Exception as e:  # 出錯不影響模型推理
                print(f"  ⚠ attention probs 計算失敗 (phase={ext._current_phase}, "
                      f"layer={ext._current_layer}): {e}")
            return att_output

        vlm_expert.eager_attention_forward = patched_eager

        print(f"  已註冊 hook (num_vlm_layers={vlm_expert.num_vlm_layers}, "
              f"heads={self.num_heads}, patch_grid={self.patch_grid}, "
              f"tokens/cam={self.tokens_per_camera})")

    def clear(self) -> None:
        self.attention_maps = {}
        self.prefix_boundaries = None
        self._current_phase = "prefix"
        self._current_layer = -1
        self._current_type = "self"
        self._denoise_step_idx = 0
        # prefix 階段收容器
        self.attention_maps["prefix"] = {}

    def remove_hooks(self) -> None:
        if self._vlm_expert is not None:
            self._vlm_expert.eager_attention_forward = self._orig_eager
            self._vlm_expert.forward_attn_layer = self._orig_self
            self._vlm_expert.forward_cross_attn_layer = self._orig_cross
        if self._model_vla is not None:
            if self._orig_embed_prefix is not None:
                self._model_vla.embed_prefix = self._orig_embed_prefix
            if self._orig_denoise_step is not None:
                self._model_vla.denoise_step = self._orig_denoise_step
        print("  已移除 hook")

    # -------- internals --------

    def _measure_patch_grid(self) -> None:
        """用 dummy image 呼叫 embed_image 精確量測 tokens_per_camera 與 grid side。"""
        vlm_expert = self._vlm_expert
        device = next(vlm_expert.parameters()).device
        img_size = vlm_expert.config.vision_config.image_size
        # SmolVLM 會做 image splitting,為了量測「單次 embed_image 輸出」我們用 do_image_splitting=False 無關
        # 這裡直接丟一張 (1,3,H,W) 的零影像量測
        dummy = torch.zeros(1, 3, img_size, img_size,
                            dtype=vlm_expert.get_vlm_model().vision_model.dtype,
                            device=device)
        with torch.no_grad():
            out = vlm_expert.embed_image(dummy)
        n = int(out.shape[1])
        self.tokens_per_camera = n
        side = int(round(n**0.5))
        if side * side != n:
            # fallback:從 config 算
            patch_size = vlm_expert.config.vision_config.patch_size
            side_cfg = img_size // patch_size
            if side_cfg * side_cfg >= n:
                side = side_cfg
            else:
                raise RuntimeError(f"無法推導 patch grid:tokens={n}, side={side}")
        self.patch_grid = (side, side)

    def _record_boundaries(self, images, lang_tokens, lang_masks, embs,
                           pad_masks, att_masks) -> None:
        """根據 embed_prefix 的拼接順序精準算出每段區間。"""
        n_cam = len(images)
        use_special = bool(self._model_vla.add_image_special_tokens)
        special_before = 1 if use_special else 0
        special_after = 1 if use_special else 0
        n_per_cam_block = special_before + self.tokens_per_camera + special_after

        cameras: OrderedDict[str, tuple[int, int]] = OrderedDict()
        cursor = 0
        for i in range(n_cam):
            patch_start = cursor + special_before
            patch_end = patch_start + self.tokens_per_camera
            # 若 camera_keys 數與 images 數一致則用其 key,否則用索引
            cam_key = self._camera_keys[i] if i < len(self._camera_keys) else f"camera_{i}"
            cameras[cam_key] = (patch_start, patch_end)
            cursor += n_per_cam_block

        lang_start = cursor
        lang_len_total = int(lang_tokens.shape[1])  # 含 padding
        lang_len_effective = int(lang_masks[0].sum().item()) if lang_masks is not None else lang_len_total
        lang_end = lang_start + lang_len_total  # 模型端以 lang_masks 過濾 padding
        cursor += lang_len_total

        # state
        state_start = cursor
        # state 目前是單一 token(state_proj 投影後再 None 擴維 → seq=1)
        states_seq_len = 1
        state_end = state_start + states_seq_len

        seq_len = state_end
        padded_len = int(embs.shape[1])

        self.prefix_boundaries = {
            "cameras": cameras,
            "language": (lang_start, lang_end),
            "language_effective_len": lang_len_effective,
            "state": (state_start, state_end),
            "seq_len": seq_len,
            "padded_len": padded_len,
        }

    def _compute_probs(self, attention_mask, head_dim,
                       query_states, key_states) -> torch.Tensor:
        """重現 eager_attention_forward 內 softmax,回傳 [B, heads, Q, K] fp32 on CPU 轉 store_dtype。"""
        num_att = self._vlm_expert.num_attention_heads
        num_kv = self._vlm_expert.num_key_value_heads
        num_grp = num_att // num_kv
        seq_k = key_states.shape[1]
        batch = key_states.shape[0]

        k_exp = key_states[:, :, :, None, :].expand(
            batch, seq_k, num_kv, num_grp, head_dim
        ).reshape(batch, seq_k, num_att, head_dim)

        q = query_states.to(torch.float32).transpose(1, 2)
        k = k_exp.to(torch.float32).transpose(1, 2)
        raw = torch.matmul(q, k.transpose(2, 3)) * (head_dim**-0.5)
        mask_val = torch.finfo(raw.dtype).min
        raw = torch.where(attention_mask[:, None, :, :].bool(), raw, mask_val)
        probs = F.softmax(raw, dim=-1)
        return probs.detach().to(self._store_dtype).cpu()

    def _store_probs(self, probs: torch.Tensor) -> None:
        """probs shape = [B, heads, Q, K],我們存 B=0 切片。"""
        phase = self._current_phase
        layer = self._current_layer
        kind = self._current_type
        if phase not in self.attention_maps:
            self.attention_maps[phase] = {}
        p = probs[0]  # [heads, Q, K]
        self.attention_maps[phase][layer] = {
            "type": kind,
            "probs": p,
            "q_len": int(p.shape[1]),
            "k_len": int(p.shape[2]),
        }


# ============================================================
# 2. 輸入轉 batch
# ============================================================


def prepare_batch_from_frame(model, frame: dict, task_description: str, device) -> dict:
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    batch = {}
    for key in model.config.image_features:
        if key in frame:
            img = frame[key]
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            batch[key] = img.to(device)

    if OBS_STATE in frame:
        state = frame[OBS_STATE]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch[OBS_STATE] = state.float().to(device)

    tokenizer = model.model.vlm_with_expert.processor.tokenizer
    tokenized = tokenizer(
        task_description, return_tensors="pt",
        padding=False, truncation=True, max_length=128,
    )
    batch[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].bool().to(device)
    return batch


# ============================================================
# 3. 切片工具 (下游可視化唯一使用這裡取資料)
# ============================================================


def get_vision_attn_1d(
    ext: AttentionExtractor,
    phase: str,
    layer_idx: int,
    camera_key: str,
    *,
    head: int | None = None,
    query_range: slice | int | None = None,
) -> np.ndarray:
    """
    從 attention_maps[phase][layer] 取出給定相機的 1D vision attention 向量。

    Args:
        head         : 指定 head;None 時對所有 head 取平均
        query_range  : 指定 action token 範圍;None 時對所有 Q 取平均
    """
    bdry = ext.prefix_boundaries
    if bdry is None:
        raise RuntimeError("尚未擷取 prefix_boundaries")
    if camera_key not in bdry["cameras"]:
        raise KeyError(f"{camera_key} 不在 prefix 內,可用:{list(bdry['cameras'])}")
    v_s, v_e = bdry["cameras"][camera_key]

    info = ext.attention_maps[phase][layer_idx]
    probs = info["probs"]  # [heads, Q, K]
    k_len = probs.shape[-1]
    v_e_clip = min(v_e, k_len)

    if head is None:
        arr = probs[:, :, v_s:v_e_clip].float().mean(dim=0)
    else:
        arr = probs[head, :, v_s:v_e_clip].float()
    # Q 維度
    if query_range is None:
        arr = arr.mean(dim=0)
    elif isinstance(query_range, int):
        arr = arr[query_range]
    else:
        arr = arr[query_range].mean(dim=0)
    return arr.numpy()


def get_modality_sums(
    ext: AttentionExtractor,
    phase: str,
    layer_idx: int,
) -> dict[str, float]:
    """回傳該層 attention 對 vision / language / state 各模態的總量(先平均 heads 與 Q)。"""
    bdry = ext.prefix_boundaries
    probs = ext.attention_maps[phase][layer_idx]["probs"].float()
    attn_k = probs.mean(dim=0).mean(dim=0).numpy()  # [K]
    k_len = attn_k.shape[0]

    v_sum = 0.0
    for _cam, (s, e) in bdry["cameras"].items():
        v_sum += float(attn_k[s:min(e, k_len)].sum())
    ls, le = bdry["language"]
    # 只取 effective 長度(排除 padding)
    le_eff = ls + bdry.get("language_effective_len", le - ls)
    l_sum = float(attn_k[ls:min(le_eff, k_len)].sum())
    ss, se = bdry["state"]
    s_sum = float(attn_k[ss:min(se, k_len)].sum())
    return {"vision": v_sum, "language": l_sum, "state": s_sum}


# ============================================================
# 4. 可視化小工具
# ============================================================


def _attn_to_heatmap(attn_1d: np.ndarray, grid_hw: tuple[int, int],
                     target_hw: tuple[int, int],
                     vmax: float | None = None) -> tuple[np.ndarray, float]:
    h, w = grid_hw
    assert len(attn_1d) == h * w, f"patch grid mismatch: {len(attn_1d)} != {h}*{w}"
    grid = attn_1d.reshape(h, w).astype(np.float32)
    if vmax is None:
        vmax = float(grid.max() + 1e-9)
        norm = grid / vmax
    else:
        norm = np.clip(grid / (vmax + 1e-9), 0, 1)
    heatmap = cv2.resize(norm, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_LINEAR)
    return heatmap, vmax


def overlay_heatmap(image_rgb: np.ndarray, heatmap01: np.ndarray,
                    base_alpha: float = 0.75) -> np.ndarray:
    """把 [0,1] heatmap 以 hot colormap 疊加,alpha 與 attention 強度成比例(避免全圖塗色)。"""
    colored = (cm.hot(heatmap01)[:, :, :3] * 255).astype(np.uint8)
    alpha = (base_alpha * heatmap01)[:, :, None]
    blended = image_rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def image_to_rgb_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


# ============================================================
# 5. 靜態單幀可視化(多維度)
# ============================================================


def visualize_frame_comprehensive(
    ext: AttentionExtractor,
    frame_imgs: dict[str, np.ndarray],
    lang_token_strs: list[str],
    frame_idx: int,
    output_dir: Path,
    *,
    last_cross_layer: int | None = None,
) -> None:
    """
    對單一幀輸出多張 PNG,涵蓋所有可視化維度。

    - 01_cameras_overlay_last_cross.png  : 每台相機 × 最後一層 cross-attn 疊加
    - 02_denoise_timeline.png            : 最後一層 cross-attn 在 10 步去噪上的熱圖(camera1)
    - 03_per_head_last_cross.png         : 最後一層 cross-attn 每 head 熱圖
    - 04_per_action_token.png            : 代表性 action token (0,12,24,36,49) 的 heatmap
    - 05_lang_bar.png                    : 語言 token attention (最後一層 cross)
    - 06_modality_by_layer.png           : 各去噪步 × 各層的 v/l/s 佔比折線
    - 07_expert_self_attn.png            : 最後一步 expert self-attn 50×50 矩陣
    - 08_prefix_self_attn.png            : prefix 最後一層 self-attn (vision×vision 區塊)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    denoise_phases = sorted([p for p in ext.attention_maps if p.startswith("denoise_")])
    if not denoise_phases:
        print(f"  ⚠ frame {frame_idx}:沒有 denoise 階段資料")
        return
    last_phase = denoise_phases[-1]

    # 取最後一個 cross-attn 層 idx
    if last_cross_layer is None:
        cross_layers = [i for i, v in ext.attention_maps[last_phase].items() if v["type"] == "cross"]
        if not cross_layers:
            print(f"  ⚠ frame {frame_idx}:找不到 cross-attn 層")
            return
        last_cross_layer = max(cross_layers)

    cam_keys = list(ext.prefix_boundaries["cameras"].keys())
    grid_hw = ext.patch_grid

    # ── 01. 每台相機疊加 ─────────────────────────────────
    n_cam = len(cam_keys)
    fig, axes = plt.subplots(2, max(n_cam, 1), figsize=(5 * max(n_cam, 1), 9), squeeze=False)
    for i, cam in enumerate(cam_keys):
        img = frame_imgs.get(cam)
        if img is None:
            axes[0, i].axis("off")
            axes[1, i].axis("off")
            continue
        attn = get_vision_attn_1d(ext, last_phase, last_cross_layer, cam)
        hm, _ = _attn_to_heatmap(attn, grid_hw, img.shape[:2])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{cam}\n原圖", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(overlay_heatmap(img, hm))
        axes[1, i].set_title(f"{cam}\n{last_phase} / layer {last_cross_layer} cross",
                             fontsize=10)
        axes[1, i].axis("off")
    fig.suptitle(f"Frame {frame_idx}  多相機 cross-attention 疊加", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "01_cameras_overlay_last_cross.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── 02. denoise timeline(cam1 / last cross layer) ─────
    primary_cam = cam_keys[0]
    img_primary = frame_imgs.get(primary_cam)
    if img_primary is not None:
        n_steps = len(denoise_phases)
        cols = min(n_steps, 5)
        rows = (n_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
        for si, ph in enumerate(denoise_phases):
            r, c = si // cols, si % cols
            if last_cross_layer not in ext.attention_maps[ph]:
                axes[r, c].axis("off")
                continue
            attn = get_vision_attn_1d(ext, ph, last_cross_layer, primary_cam)
            hm, _ = _attn_to_heatmap(attn, grid_hw, img_primary.shape[:2])
            axes[r, c].imshow(overlay_heatmap(img_primary, hm))
            axes[r, c].set_title(ph, fontsize=10)
            axes[r, c].axis("off")
        for si in range(n_steps, rows * cols):
            axes[si // cols, si % cols].axis("off")
        fig.suptitle(f"Frame {frame_idx}  去噪 10 步 × cross-attention (cam={primary_cam}, "
                     f"layer={last_cross_layer})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "02_denoise_timeline.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # ── 03. per-head ─────────────────────────────────
    if img_primary is not None and last_cross_layer in ext.attention_maps[last_phase]:
        n_heads = ext.num_heads
        cols = min(n_heads, 5)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows), squeeze=False)
        for hi in range(n_heads):
            r, c = hi // cols, hi % cols
            attn = get_vision_attn_1d(ext, last_phase, last_cross_layer, primary_cam, head=hi)
            hm, _ = _attn_to_heatmap(attn, grid_hw, img_primary.shape[:2])
            axes[r, c].imshow(overlay_heatmap(img_primary, hm))
            axes[r, c].set_title(f"head {hi}", fontsize=9)
            axes[r, c].axis("off")
        for hi in range(n_heads, rows * cols):
            axes[hi // cols, hi % cols].axis("off")
        fig.suptitle(f"Frame {frame_idx}  {last_phase} / layer {last_cross_layer} per-head",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "03_per_head_last_cross.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # ── 04. per action token ─────────────────────────
    chunk_size = ext.attention_maps[last_phase][last_cross_layer]["q_len"]
    samples = sorted(set([0, chunk_size // 4, chunk_size // 2, 3 * chunk_size // 4, chunk_size - 1]))
    if img_primary is not None:
        fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 3.5), squeeze=False)
        for i, tk in enumerate(samples):
            attn = get_vision_attn_1d(ext, last_phase, last_cross_layer, primary_cam,
                                      query_range=int(tk))
            hm, _ = _attn_to_heatmap(attn, grid_hw, img_primary.shape[:2])
            axes[0, i].imshow(overlay_heatmap(img_primary, hm))
            axes[0, i].set_title(f"action token {tk}", fontsize=10)
            axes[0, i].axis("off")
        fig.suptitle(f"Frame {frame_idx}  不同 action token 的 vision attention "
                     f"({last_phase} / layer {last_cross_layer})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "04_per_action_token.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # ── 05. language bar ─────────────────────────────
    info = ext.attention_maps[last_phase][last_cross_layer]
    probs = info["probs"].float().mean(dim=0).mean(dim=0).numpy()  # [K]
    ls, le = ext.prefix_boundaries["language"]
    le_eff = ls + ext.prefix_boundaries.get("language_effective_len", le - ls)
    le_eff = min(le_eff, probs.shape[0])
    l_attn = probs[ls:le_eff]
    labels = [t.replace("▁", "") for t in lang_token_strs[: len(l_attn)]]
    fig, ax = plt.subplots(figsize=(max(6, len(l_attn) * 0.6), 4))
    ax.bar(range(len(l_attn)), l_attn, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(l_attn)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(f"語言 token attention  {last_phase} / layer {last_cross_layer}")
    ax.set_ylabel("平均 attention weight")
    fig.tight_layout()
    fig.savefig(output_dir / "05_lang_bar.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── 06. modality × step × layer 折線 ─────────────
    all_layers = sorted(ext.attention_maps[last_phase].keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    cmap = plt.get_cmap("viridis", len(denoise_phases))
    for si, ph in enumerate(denoise_phases):
        vs, ls_, ss_ = [], [], []
        for li in all_layers:
            if li not in ext.attention_maps[ph]:
                vs.append(np.nan); ls_.append(np.nan); ss_.append(np.nan); continue
            m = get_modality_sums(ext, ph, li)
            tot = m["vision"] + m["language"] + m["state"] + 1e-9
            vs.append(m["vision"] / tot)
            ls_.append(m["language"] / tot)
            ss_.append(m["state"] / tot)
        color = cmap(si)
        axes[0].plot(all_layers, vs, color=color, label=ph, alpha=0.85)
        axes[1].plot(all_layers, ls_, color=color, label=ph, alpha=0.85)
        axes[2].plot(all_layers, ss_, color=color, label=ph, alpha=0.85)
    for a, t in zip(axes, ["視覺佔比", "語言佔比", "狀態佔比"]):
        a.set_title(t); a.set_xlabel("layer_idx"); a.grid(alpha=0.3)
    axes[0].set_ylabel("佔比")
    axes[2].legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Frame {frame_idx}  各去噪步 × 每層 模態佔比", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "06_modality_by_layer.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── 07. expert self-attention (Q=K=action tokens) ─
    self_layers = [i for i, v in ext.attention_maps[last_phase].items() if v["type"] == "self"]
    if self_layers:
        sl = max(self_layers)
        probs_sa = ext.attention_maps[last_phase][sl]["probs"].float().mean(dim=0).numpy()
        # Q 是 action tokens (後 chunk_size 個),K 是 [cached prefix + suffix]。
        # 取 Q×(K 的後 chunk_size 個) → 看 action token 之間的互相關注
        q_len = probs_sa.shape[0]
        k_len = probs_sa.shape[1]
        if k_len >= q_len:
            mat = probs_sa[:, k_len - q_len:]  # Q × Q 等長
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_title(f"Expert self-attention (action×action)  "
                         f"{last_phase} / layer {sl}")
            ax.set_xlabel("K action token")
            ax.set_ylabel("Q action token")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(output_dir / "07_expert_self_attn.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

    # ── 08. prefix self-attention ─────────────────────
    prefix_layers = sorted(ext.attention_maps.get("prefix", {}).keys())
    if prefix_layers:
        pl = prefix_layers[-1]
        probs_pref = ext.attention_maps["prefix"][pl]["probs"].float().mean(dim=0).numpy()
        seq_eff = ext.prefix_boundaries["seq_len"]
        probs_pref = probs_pref[:seq_eff, :seq_eff]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(probs_pref, cmap="magma", aspect="auto")
        # 標示模態邊界
        bdry = ext.prefix_boundaries
        for cam, (s, e) in bdry["cameras"].items():
            ax.axhline(s, color="cyan", lw=0.4, alpha=0.6)
            ax.axvline(s, color="cyan", lw=0.4, alpha=0.6)
        ls_, _ = bdry["language"]
        ss_, _ = bdry["state"]
        ax.axhline(ls_, color="yellow", lw=0.7)
        ax.axvline(ls_, color="yellow", lw=0.7)
        ax.axhline(ss_, color="lime", lw=0.7)
        ax.axvline(ss_, color="lime", lw=0.7)
        ax.set_title(f"Prefix self-attention  layer {pl}  "
                     f"(黃=language 邊界, 綠=state 邊界, 青=每相機起點)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "08_prefix_self_attn.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 6. 端對端 pipeline
# ============================================================


def _entropy(p: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def _summarize_frame(ext: AttentionExtractor, frame_idx: int,
                     lang_token_strs: list[str]) -> list[dict]:
    """回傳該幀的扁平摘要行(每行對應 phase × layer × camera)。"""
    rows: list[dict] = []
    bdry = ext.prefix_boundaries
    cam_keys = list(bdry["cameras"].keys())
    ls, le = bdry["language"]
    le_eff = ls + bdry.get("language_effective_len", le - ls)

    for phase, layers in ext.attention_maps.items():
        for layer, info in layers.items():
            probs = info["probs"].float()
            # 平均 heads 與 Q
            attn_k = probs.mean(dim=0).mean(dim=0).numpy()
            k_len = attn_k.shape[0]
            mods = get_modality_sums(ext, phase, layer)

            # 每相機 vision 統計
            cam_stats = {}
            for cam in cam_keys:
                s, e = bdry["cameras"][cam]
                e = min(e, k_len)
                v = attn_k[s:e]
                cam_stats[cam] = {
                    "sum": float(v.sum()),
                    "argmax_flat": int(v.argmax()) if v.size else -1,
                    "entropy": _entropy(v) if v.size else 0.0,
                }

            # top-3 language token
            le_clip = min(le_eff, k_len)
            l_attn = attn_k[ls:le_clip]
            top3 = []
            if l_attn.size:
                order = np.argsort(-l_attn)[:3]
                for idx in order:
                    tok = lang_token_strs[idx].replace("▁", "") if idx < len(lang_token_strs) else f"?{idx}"
                    top3.append((tok, float(l_attn[idx])))

            rows.append({
                "frame": frame_idx,
                "phase": phase,
                "layer": layer,
                "type": info["type"],
                "q_len": info["q_len"],
                "k_len": info["k_len"],
                "vision_sum": mods["vision"],
                "language_sum": mods["language"],
                "state_sum": mods["state"],
                "cameras": cam_stats,
                "lang_top3": top3,
            })
    return rows


def _write_summary(rows: list[dict], out_dir: Path) -> None:
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if not rows:
        return
    cam_keys = list(rows[0]["cameras"].keys())
    fieldnames = ["frame", "phase", "layer", "type", "q_len", "k_len",
                  "vision_sum", "language_sum", "state_sum",
                  "lang_top1", "lang_top1_w"]
    for cam in cam_keys:
        fieldnames += [f"{cam}__sum", f"{cam}__argmax", f"{cam}__entropy"]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in ["frame", "phase", "layer", "type",
                                      "q_len", "k_len", "vision_sum",
                                      "language_sum", "state_sum"]}
            top = r["lang_top3"][0] if r["lang_top3"] else ("", 0.0)
            out["lang_top1"] = top[0]
            out["lang_top1_w"] = top[1]
            for cam in cam_keys:
                c = r["cameras"][cam]
                out[f"{cam}__sum"] = c["sum"]
                out[f"{cam}__argmax"] = c["argmax_flat"]
                out[f"{cam}__entropy"] = c["entropy"]
            w.writerow(out)


def extract_attention_from_episode(
    model,
    dataset,
    frame_indices: list[int],
    task_description: str,
    *,
    camera_keys: list[str] | None = None,
    output_dir: str | Path = "attention_analysis",
    device=None,
    make_per_frame_png: bool = True,
) -> None:
    """
    對指定 frames 做推理,提取 cross/self-attention,輸出:
      - 每幀 8 張 PNG (make_per_frame_png=True 時)
      - summary.json / summary.csv
    """
    if device is None:
        device = next(model.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if camera_keys is None:
        camera_keys = list(model.config.image_features)

    ext = AttentionExtractor()
    ext.register_hooks(model)
    model.eval()

    tokenizer = model.model.vlm_with_expert.processor.tokenizer
    _tok = tokenizer(task_description, return_tensors="pt", padding=False)
    lang_token_strs = tokenizer.convert_ids_to_tokens(_tok["input_ids"][0].tolist())

    all_rows: list[dict] = []
    with torch.no_grad():
        for idx in frame_indices:
            ext.clear()
            model.reset()
            frame = dataset[idx]
            batch = prepare_batch_from_frame(model, frame, task_description, device)
            try:
                _ = model.predict_action_chunk(batch)
            except Exception as e:
                print(f"  ⚠ frame {idx} 推理失敗: {e}")
                continue
            if ext.prefix_boundaries is None:
                print(f"  ⚠ frame {idx} 未擷取到 boundaries")
                continue

            # 收集各相機圖像
            frame_imgs = {}
            for cam in camera_keys:
                t = frame.get(cam)
                if t is not None:
                    frame_imgs[cam] = image_to_rgb_uint8(t)

            if make_per_frame_png:
                per_dir = output_dir / f"frame_{idx:06d}"
                visualize_frame_comprehensive(
                    ext, frame_imgs, lang_token_strs, idx, per_dir,
                )

            all_rows.extend(_summarize_frame(ext, idx, lang_token_strs))
            print(f"  [OK] frame {idx}: phases={len(ext.attention_maps)} "
                  f"layers/phase={[len(v) for v in ext.attention_maps.values()]}")

    _write_summary(all_rows, output_dir)
    ext.remove_hooks()
    print(f"\n[完成] 結果在 {output_dir}/ (summary.json, summary.csv)")


# ============================================================
# 執行範例
# ============================================================


if __name__ == "__main__":
    MODEL_PATH = "wuc1/bi_so101_flatten-and-fold-the-rag-32layers-smolvla_base-0403-model"
    DATASET_REPO = "wuc1/bi_so101_flatten-and-fold-the-rag-0331"
    TASK_DESCRIPTION = "整平並摺疊毛巾"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置:{device}")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model = SmolVLAPolicy.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(DATASET_REPO, video_backend="pyav")

    print(f"模型 attention_mode: {model.model.vlm_with_expert.attention_mode}")
    print(f"self_attn_every_n_layers: {model.model.vlm_with_expert.self_attn_every_n_layers}")

    key_frames = [2130]
    extract_attention_from_episode(
        model, dataset,
        frame_indices=key_frames,
        task_description=TASK_DESCRIPTION,
        output_dir="attention_analysis",
        device=device,
    )
