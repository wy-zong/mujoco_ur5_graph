"""
SmolVLA Attention 影片可視化工具
================================
使用 `extract_attention_maps.AttentionExtractor` 提取每幀的完整 attention
(10 步去噪 × 16 層 × 多相機 × 多 head × 50 action tokens),並輸出 MP4。

支援兩種版面::

    layout="basic"   舊版行為(單相機、最後一層 cross、語言長條 + 模態圓餅)
    layout="full"    完整版:
        Row 1: 每台相機 原圖 + cross-attention 疊加
        Row 2: 10 步去噪對同一相機的 cross-attention 疊加
        Row 3: per-head (最後一層 cross) 熱圖小圖
        Row 4: 代表性 action token 的 vision attention
        Row 5: 語言長條 / 模態圓餅 / 去噪步 × 模態折線

支援 `global_norm=True`:先掃一次所有 frames 記錄最大 attention 值,
第二次寫影片時統一 colormap 範圍 → 跨幀強度可比較。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")

try:
    import matplotlib.font_manager as fm

    for _n in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
        if _n in {f.name for f in fm.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = _n
            break
except Exception:
    pass

from extract_attention_maps import (
    AttentionExtractor,
    get_modality_sums,
    get_vision_attn_1d,
    image_to_rgb_uint8,
    overlay_heatmap,
    prepare_batch_from_frame,
)


# ============================================================
# 工具
# ============================================================


def _attn1d_to_heatmap(attn_1d: np.ndarray, grid_hw: tuple[int, int],
                       target_hw: tuple[int, int],
                       vmax: float | None = None) -> np.ndarray:
    """回傳 [0,1] heatmap。vmax 為 None 時以本幀 max 做 normalization。"""
    h, w = grid_hw
    assert len(attn_1d) == h * w, f"patch grid mismatch: {len(attn_1d)} vs {h}*{w}"
    grid = attn_1d.reshape(h, w).astype(np.float32)
    if vmax is None:
        vmax = float(grid.max() + 1e-9)
    norm = np.clip(grid / (vmax + 1e-9), 0, 1)
    return cv2.resize(norm, (target_hw[1], target_hw[0]),
                      interpolation=cv2.INTER_LINEAR)


def _fig_to_rgb(fig, width: int, height: int) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return cv2.resize(buf, (width, height))


# ============================================================
# Row 渲染器
# ============================================================


def _row_cameras(ext: AttentionExtractor, frame_imgs: dict[str, np.ndarray],
                 phase: str, layer: int, row_h: int, row_w: int,
                 vmax_per_cam: dict[str, float] | None = None) -> np.ndarray:
    cams = list(ext.prefix_boundaries["cameras"].keys())
    n = max(len(cams), 1)
    tile_w = row_w // n
    tiles = []
    for cam in cams:
        img = frame_imgs.get(cam)
        if img is None:
            tiles.append(np.zeros((row_h, tile_w, 3), dtype=np.uint8))
            continue
        # 左:原圖縮放;右:疊加(同寬)。這裡用整格疊加以節省橫向空間
        attn = get_vision_attn_1d(ext, phase, layer, cam)
        vmax = vmax_per_cam.get(cam) if vmax_per_cam else None
        hm = _attn1d_to_heatmap(attn, ext.patch_grid, img.shape[:2], vmax=vmax)
        ovl = overlay_heatmap(img, hm)
        # 頂部放相機名稱
        cv2.putText(ovl, cam, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tile = cv2.resize(ovl, (tile_w, row_h), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    out = np.hstack(tiles) if tiles else np.zeros((row_h, row_w, 3), dtype=np.uint8)
    if out.shape[1] != row_w:
        pad = np.zeros((row_h, row_w - out.shape[1], 3), dtype=np.uint8)
        out = np.hstack([out, pad])
    return out


def _row_denoise_timeline(ext: AttentionExtractor, img: np.ndarray,
                          cam: str, layer: int,
                          row_h: int, row_w: int,
                          vmax: float | None = None) -> np.ndarray:
    phases = sorted([p for p in ext.attention_maps if p.startswith("denoise_")])
    n = max(len(phases), 1)
    tile_w = row_w // n
    tiles = []
    for ph in phases:
        if layer not in ext.attention_maps.get(ph, {}):
            tiles.append(np.zeros((row_h, tile_w, 3), dtype=np.uint8))
            continue
        attn = get_vision_attn_1d(ext, ph, layer, cam)
        hm = _attn1d_to_heatmap(attn, ext.patch_grid, img.shape[:2], vmax=vmax)
        ovl = overlay_heatmap(img, hm)
        cv2.putText(ovl, ph.replace("denoise_", "step "), (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(ovl, (tile_w, row_h), interpolation=cv2.INTER_AREA))
    out = np.hstack(tiles) if tiles else np.zeros((row_h, row_w, 3), dtype=np.uint8)
    if out.shape[1] != row_w:
        pad = np.zeros((row_h, row_w - out.shape[1], 3), dtype=np.uint8)
        out = np.hstack([out, pad])
    return out


def _row_per_head(ext: AttentionExtractor, img: np.ndarray, cam: str,
                  phase: str, layer: int, row_h: int, row_w: int,
                  vmax: float | None = None) -> np.ndarray:
    n_heads = ext.num_heads
    tile_w = row_w // max(n_heads, 1)
    tiles = []
    for hi in range(n_heads):
        attn = get_vision_attn_1d(ext, phase, layer, cam, head=hi)
        hm = _attn1d_to_heatmap(attn, ext.patch_grid, img.shape[:2], vmax=vmax)
        ovl = overlay_heatmap(img, hm, base_alpha=0.7)
        cv2.putText(ovl, f"h{hi}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(ovl, (tile_w, row_h), interpolation=cv2.INTER_AREA))
    out = np.hstack(tiles)
    if out.shape[1] != row_w:
        pad = np.zeros((row_h, row_w - out.shape[1], 3), dtype=np.uint8)
        out = np.hstack([out, pad])
    return out


def _row_per_token(ext: AttentionExtractor, img: np.ndarray, cam: str,
                   phase: str, layer: int, row_h: int, row_w: int,
                   vmax: float | None = None) -> np.ndarray:
    chunk = ext.attention_maps[phase][layer]["q_len"]
    samples = sorted(set([0, chunk // 4, chunk // 2, 3 * chunk // 4, chunk - 1]))
    tile_w = row_w // len(samples)
    tiles = []
    for tk in samples:
        attn = get_vision_attn_1d(ext, phase, layer, cam, query_range=int(tk))
        hm = _attn1d_to_heatmap(attn, ext.patch_grid, img.shape[:2], vmax=vmax)
        ovl = overlay_heatmap(img, hm)
        cv2.putText(ovl, f"tok{tk}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(ovl, (tile_w, row_h), interpolation=cv2.INTER_AREA))
    out = np.hstack(tiles)
    if out.shape[1] != row_w:
        pad = np.zeros((row_h, row_w - out.shape[1], 3), dtype=np.uint8)
        out = np.hstack([out, pad])
    return out


def _row_stats(ext: AttentionExtractor, lang_token_strs: list[str],
               phase: str, layer: int,
               row_h: int, row_w: int) -> np.ndarray:
    """語言長條 | 模態圓餅 | 去噪步 × 模態折線"""
    info = ext.attention_maps[phase][layer]
    attn_k = info["probs"].float().mean(dim=0).mean(dim=0).numpy()
    k_len = attn_k.shape[0]

    # 語言長條
    ls, le = ext.prefix_boundaries["language"]
    le_eff = ls + ext.prefix_boundaries.get("language_effective_len", le - ls)
    l_attn = attn_k[ls:min(le_eff, k_len)]
    labels = [t.replace("▁", "") for t in lang_token_strs[: len(l_attn)]]
    w_each = row_w // 3
    dpi = 100

    fig1, ax1 = plt.subplots(figsize=(w_each / dpi, row_h / dpi), dpi=dpi)
    ax1.bar(range(len(l_attn)), l_attn, color="steelblue", alpha=0.85)
    ax1.set_xticks(range(len(l_attn)))
    ax1.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax1.set_title(f"Language attn  {phase}/L{layer}", fontsize=9)
    fig1.tight_layout(pad=0.3)
    part1 = _fig_to_rgb(fig1, w_each, row_h)

    # 模態圓餅
    mods = get_modality_sums(ext, phase, layer)
    tot = mods["vision"] + mods["language"] + mods["state"] + 1e-9
    fig2, ax2 = plt.subplots(figsize=(w_each / dpi, row_h / dpi), dpi=dpi)
    ax2.pie(
        [mods["vision"] / tot, mods["language"] / tot, mods["state"] / tot],
        labels=[f"視覺\n{mods['vision']/tot*100:.1f}%",
                f"語言\n{mods['language']/tot*100:.1f}%",
                f"狀態\n{mods['state']/tot*100:.1f}%"],
        colors=["#ff7f7f", "#7fbfff", "#7fff7f"],
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    ax2.set_title(f"模態佔比  {phase}/L{layer}", fontsize=9)
    fig2.tight_layout(pad=0.3)
    part2 = _fig_to_rgb(fig2, w_each, row_h)

    # 去噪步 × 模態折線
    phases = sorted([p for p in ext.attention_maps if p.startswith("denoise_")])
    fig3, ax3 = plt.subplots(figsize=(w_each / dpi, row_h / dpi), dpi=dpi)
    vs, lss, sss = [], [], []
    for ph in phases:
        if layer in ext.attention_maps.get(ph, {}):
            m = get_modality_sums(ext, ph, layer)
            t = m["vision"] + m["language"] + m["state"] + 1e-9
            vs.append(m["vision"] / t); lss.append(m["language"] / t); sss.append(m["state"] / t)
        else:
            vs.append(np.nan); lss.append(np.nan); sss.append(np.nan)
    x = np.arange(len(phases))
    ax3.plot(x, vs, "-o", color="#ff4f4f", label="視覺", markersize=3)
    ax3.plot(x, lss, "-o", color="#4f7fff", label="語言", markersize=3)
    ax3.plot(x, sss, "-o", color="#4fff4f", label="狀態", markersize=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{i}" for i in range(len(phases))], fontsize=7)
    ax3.set_xlabel("denoise step", fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=7, loc="center right")
    ax3.set_title(f"各去噪步佔比  (L{layer})", fontsize=9)
    ax3.grid(alpha=0.3)
    fig3.tight_layout(pad=0.3)
    part3 = _fig_to_rgb(fig3, w_each, row_h)

    out = np.hstack([part1, part2, part3])
    if out.shape[1] != row_w:
        out = cv2.resize(out, (row_w, row_h), interpolation=cv2.INTER_AREA)
    return out


# ============================================================
# 版面合成
# ============================================================


def _pick_last_cross_layer(ext: AttentionExtractor, phase: str) -> int | None:
    cross_layers = [i for i, v in ext.attention_maps.get(phase, {}).items()
                    if v["type"] == "cross"]
    return max(cross_layers) if cross_layers else None


def _compose_full(ext: AttentionExtractor, frame_imgs: dict[str, np.ndarray],
                  frame_idx: int, lang_token_strs: list[str],
                  vmax_per_cam: dict[str, float] | None = None) -> np.ndarray | None:
    denoise_phases = sorted([p for p in ext.attention_maps if p.startswith("denoise_")])
    if not denoise_phases:
        return None
    last_phase = denoise_phases[-1]
    last_layer = _pick_last_cross_layer(ext, last_phase)
    if last_layer is None:
        return None
    cams = list(ext.prefix_boundaries["cameras"].keys())
    primary_cam = cams[0] if cams else None
    if primary_cam is None or primary_cam not in frame_imgs:
        return None
    primary_img = frame_imgs[primary_cam]

    full_w = 1600
    row_heights = {
        "cameras": 300,
        "timeline": 240,
        "heads": 240,
        "tokens": 240,
        "stats": 280,
    }
    rows = []
    rows.append(_row_cameras(ext, frame_imgs, last_phase, last_layer,
                             row_heights["cameras"], full_w, vmax_per_cam))
    vmax_primary = vmax_per_cam.get(primary_cam) if vmax_per_cam else None
    rows.append(_row_denoise_timeline(ext, primary_img, primary_cam, last_layer,
                                      row_heights["timeline"], full_w, vmax_primary))
    rows.append(_row_per_head(ext, primary_img, primary_cam, last_phase, last_layer,
                              row_heights["heads"], full_w, vmax_primary))
    rows.append(_row_per_token(ext, primary_img, primary_cam, last_phase, last_layer,
                               row_heights["tokens"], full_w, vmax_primary))
    rows.append(_row_stats(ext, lang_token_strs, last_phase, last_layer,
                           row_heights["stats"], full_w))

    frame = np.vstack(rows)
    banner = np.zeros((30, full_w, 3), dtype=np.uint8)
    cv2.putText(banner,
                f"Frame {frame_idx}  last cross layer={last_layer}  cam_primary={primary_cam}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    frame = np.vstack([banner, frame])
    return frame


def _compose_basic(ext: AttentionExtractor, frame_imgs: dict[str, np.ndarray],
                   frame_idx: int, lang_token_strs: list[str],
                   vmax_per_cam: dict[str, float] | None = None) -> np.ndarray | None:
    denoise_phases = sorted([p for p in ext.attention_maps if p.startswith("denoise_")])
    if not denoise_phases:
        return None
    last_phase = denoise_phases[-1]
    last_layer = _pick_last_cross_layer(ext, last_phase)
    if last_layer is None:
        return None
    cams = list(ext.prefix_boundaries["cameras"].keys())
    cam = cams[0]
    img = frame_imgs.get(cam)
    if img is None:
        return None
    h, w = img.shape[:2]

    attn = get_vision_attn_1d(ext, last_phase, last_layer, cam)
    vmax = vmax_per_cam.get(cam) if vmax_per_cam else None
    hm = _attn1d_to_heatmap(attn, ext.patch_grid, (h, w), vmax=vmax)
    ovl = overlay_heatmap(img, hm)
    cv2.putText(ovl, f"Frame {frame_idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    stats = _row_stats(ext, lang_token_strs, last_phase, last_layer,
                       h, 3 * w)
    stats_resized = cv2.resize(stats, (w, h), interpolation=cv2.INTER_AREA)
    return np.hstack([img, ovl, stats_resized])


# ============================================================
# 主流程
# ============================================================


def _scan_vmax(model, dataset, frame_indices: list[int], task_description: str,
               device) -> dict[str, float]:
    """第一遍掃描:取得每相機最後一層 cross-attn 在全 episode 中的最大值。"""
    ext = AttentionExtractor()
    ext.register_hooks(model)
    model.eval()
    vmax: dict[str, float] = {}
    try:
        with torch.no_grad():
            for idx in tqdm(frame_indices, desc="掃描 vmax (pass 1)"):
                ext.clear(); model.reset()
                frame = dataset[idx]
                batch = prepare_batch_from_frame(model, frame, task_description, device)
                try:
                    _ = model.predict_action_chunk(batch)
                except Exception:
                    continue
                if ext.prefix_boundaries is None:
                    continue
                denoise_phases = sorted([p for p in ext.attention_maps if p.startswith("denoise_")])
                if not denoise_phases:
                    continue
                last_phase = denoise_phases[-1]
                last_layer = _pick_last_cross_layer(ext, last_phase)
                if last_layer is None:
                    continue
                for cam in ext.prefix_boundaries["cameras"]:
                    attn = get_vision_attn_1d(ext, last_phase, last_layer, cam)
                    v = float(np.asarray(attn).max())
                    if v > vmax.get(cam, 0.0):
                        vmax[cam] = v
    finally:
        ext.remove_hooks()
    return vmax


def generate_attention_video(
    model,
    dataset,
    frame_indices: list[int],
    task_description: str,
    *,
    camera_keys: list[str] | None = None,
    output_path: str | Path = "attention_video.mp4",
    fps: int = 10,
    device=None,
    layout: str = "full",
    global_norm: bool = False,
) -> None:
    assert layout in {"basic", "full"}, f"layout 必須為 basic 或 full,不能是 {layout}"
    if device is None:
        device = next(model.parameters()).device
    if camera_keys is None:
        camera_keys = list(model.config.image_features)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vmax_per_cam: dict[str, float] | None = None
    if global_norm:
        vmax_per_cam = _scan_vmax(model, dataset, frame_indices, task_description, device)
        print(f"  global_norm vmax: {vmax_per_cam}")

    ext = AttentionExtractor()
    ext.register_hooks(model)
    model.eval()

    tokenizer = model.model.vlm_with_expert.processor.tokenizer
    _tok = tokenizer(task_description, return_tensors="pt", padding=False)
    lang_token_strs = tokenizer.convert_ids_to_tokens(_tok["input_ids"][0].tolist())

    writer = None
    skipped = 0
    try:
        with torch.no_grad():
            for idx in tqdm(frame_indices, desc=f"生成影片 layout={layout}"):
                ext.clear(); model.reset()
                frame = dataset[idx]
                batch = prepare_batch_from_frame(model, frame, task_description, device)
                try:
                    _ = model.predict_action_chunk(batch)
                except Exception as e:
                    print(f"\n  ⚠ frame {idx} 推理失敗: {e}")
                    skipped += 1
                    continue
                if ext.prefix_boundaries is None:
                    skipped += 1
                    continue

                frame_imgs = {}
                for cam in camera_keys:
                    t = frame.get(cam)
                    if t is not None:
                        frame_imgs[cam] = image_to_rgb_uint8(t)

                if layout == "full":
                    composed = _compose_full(ext, frame_imgs, idx, lang_token_strs, vmax_per_cam)
                else:
                    composed = _compose_basic(ext, frame_imgs, idx, lang_token_strs, vmax_per_cam)
                if composed is None:
                    skipped += 1
                    continue

                if writer is None:
                    vh, vw = composed.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (vw, vh))
                    if not writer.isOpened():
                        raise RuntimeError(f"無法開啟影片寫入器:{output_path}")
                    print(f"影片尺寸: {vw}x{vh}  FPS: {fps}  總幀數: {len(frame_indices)}")

                writer.write(cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        ext.remove_hooks()

    print(f"\n[完成] 影片儲存至: {output_path}  (跳過 {skipped} 幀)")


# ============================================================
# Episode 工具
# ============================================================


def _build_episode_index(dataset):
    ep_arr = np.array([int(e) for e in dataset.hf_dataset["episode_index"]])
    boundaries = {}
    for ep in range(dataset.num_episodes):
        idxs = np.where(ep_arr == ep)[0]
        if len(idxs):
            boundaries[ep] = (int(idxs[0]), int(idxs[-1]) + 1)
    return boundaries


def get_episode_frame_indices(dataset, episode_index: int, step: int = 1) -> list[int]:
    boundaries = _build_episode_index(dataset)
    if episode_index not in boundaries:
        raise ValueError(f"episode {episode_index} 不存在")
    ep_from, ep_to = boundaries[episode_index]
    print(f"Episode {episode_index}: 全域 index {ep_from}~{ep_to-1},共 {ep_to-ep_from} 幀")
    return list(range(ep_from, ep_to, step))


def list_episodes(dataset):
    print(f"資料集共 {dataset.num_episodes} 個 episode,總幀數 {len(dataset)}:")
    for i, (fr, to) in sorted(_build_episode_index(dataset).items()):
        print(f"  Episode {i:3d} | {fr:6d}~{to-1:6d} | {to-fr} 幀")


# ============================================================
# 執行範例
# ============================================================


if __name__ == "__main__":
    MODEL_PATH = "wuc1/bi_so101_flatten-and-fold-the-rag-32layers-smolvla_base-0403-model"
    DATASET_REPO = "wuc1/bi_so101_flatten-and-fold-the-rag-0331"
    TASK_DESCRIPTION = "整平並摺疊毛巾"

    EPISODE_INDEX = 0
    STEP = 2
    LAYOUT = "full"           # "basic" | "full"
    GLOBAL_NORM = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置:{device}")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model = SmolVLAPolicy.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(DATASET_REPO, video_backend="pyav")

    list_episodes(dataset)
    frame_indices = get_episode_frame_indices(dataset, EPISODE_INDEX, step=STEP)

    generate_attention_video(
        model, dataset,
        frame_indices=frame_indices,
        task_description=TASK_DESCRIPTION,
        output_path=f"attention_ep{EPISODE_INDEX:03d}_{LAYOUT}.mp4",
        fps=10,
        device=device,
        layout=LAYOUT,
        global_norm=GLOBAL_NORM,
    )
