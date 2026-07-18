from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


@dataclass
class FeatureRecord:
    frame_idx: int
    camera_key: str
    original_rgb: np.ndarray
    model_input_rgb: np.ndarray
    vision_tokens: np.ndarray
    connector_tokens: np.ndarray
    final_visual_tokens: np.ndarray
    vision_grid: tuple[int, int]
    connector_grid: tuple[int, int]
    input_shape: tuple[int, int]
    model_input_shape: tuple[int, int]
    connector_embed_dim: int
    embed_image_max_abs_diff: float


@dataclass
class SimplePCA:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        x = tokens.astype(np.float32, copy=False) - self.mean
        return x @ self.components.T


def fit_simple_pca(tokens: np.ndarray, n_components: int = 3) -> SimplePCA:
    x = tokens.astype(np.float32, copy=False)
    mean = x.mean(axis=0, keepdims=False)
    x_centered = x - mean
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)

    components = vt[:n_components].astype(np.float32, copy=False)
    if components.shape[0] < n_components:
        pad = np.zeros((n_components - components.shape[0], components.shape[1]), dtype=np.float32)
        components = np.vstack([components, pad])

    variances = (singular_values**2) / max(x.shape[0] - 1, 1)
    total = float(variances.sum())
    ratios = np.zeros(n_components, dtype=np.float32)
    if total > 0:
        ratios[: min(n_components, variances.shape[0])] = (
            variances[:n_components] / total
        ).astype(np.float32)
    return SimplePCA(mean=mean.astype(np.float32), components=components, explained_variance_ratio=ratios)


def parse_frame_indices(raw: str) -> list[int]:
    raw = raw.strip()
    if ":" in raw:
        parts = [int(p) for p in raw.split(":") if p]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError("--frames must be comma-separated or start:stop[:step]")
        return list(range(start, stop, step))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_camera_keys(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def to_chw_float01(t: torch.Tensor) -> torch.Tensor:
    if t.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tuple(t.shape)}")
    if t.shape[0] in (1, 3):
        img = t
    elif t.shape[-1] in (1, 3):
        img = t.permute(2, 0, 1)
    else:
        raise ValueError(f"Cannot infer image channel dimension from {tuple(t.shape)}")
    img = img.detach()
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    else:
        img = img.float()
        if float(img.max()) > 1.5:
            img = img / 255.0
    return img.clamp(0.0, 1.0)


def chw_float01_to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
    img = to_chw_float01(t).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).round().clip(0, 255).astype(np.uint8)


def model_input_to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
    # SmolVLA prepare_images returns B,C,H,W in [-1,1].
    img = t[0].detach().float().cpu()
    img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    return (img * 255.0).round().clip(0, 255).astype(np.uint8)


def closest_grid(num_tokens: int, target_aspect: float | None = None) -> tuple[int, int]:
    side = int(round(num_tokens**0.5))
    if side * side == num_tokens:
        return side, side

    factors = []
    for h in range(1, int(num_tokens**0.5) + 1):
        if num_tokens % h == 0:
            w = num_tokens // h
            factors.append((h, w))
            factors.append((w, h))
    if not factors:
        return 1, num_tokens
    if target_aspect is None:
        return min(factors, key=lambda x: abs(x[0] - x[1]))
    return min(factors, key=lambda x: abs((x[1] / max(x[0], 1)) - target_aspect))


def infer_vision_grid(
    num_tokens: int,
    image_hw: tuple[int, int],
    patch_size: int | None,
) -> tuple[int, int]:
    h, w = image_hw
    if patch_size and h % patch_size == 0 and w % patch_size == 0:
        gh, gw = h // patch_size, w // patch_size
        if gh * gw == num_tokens:
            return gh, gw
    return closest_grid(num_tokens, target_aspect=w / max(h, 1))


def infer_connector_grid(
    num_tokens: int,
    raw_grid: tuple[int, int],
    scale_factor: int | None,
) -> tuple[int, int]:
    if scale_factor and scale_factor > 0:
        gh, gw = raw_grid[0] // scale_factor, raw_grid[1] // scale_factor
        if gh * gw == num_tokens:
            return gh, gw
    return closest_grid(num_tokens, target_aspect=raw_grid[1] / max(raw_grid[0], 1))


def prepare_frame_batch(policy: SmolVLAPolicy, frame: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in policy.config.image_features:
        if key not in frame:
            continue
        img = to_chw_float01(frame[key]).unsqueeze(0).to(device)
        batch[key] = img
        mask_key = f"{key}_padding_mask"
        if mask_key in frame:
            mask = frame[mask_key]
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask)
            batch[mask_key] = mask.reshape(-1).bool().to(device)
    return batch


def actual_image_order(policy: SmolVLAPolicy, batch: dict[str, torch.Tensor]) -> list[str]:
    present = [key for key in policy.config.image_features if key in batch]
    missing = [key for key in policy.config.image_features if key not in batch]
    padded = missing[: max(0, int(policy.config.empty_cameras))]
    return present + padded


def extract_records(
    policy: SmolVLAPolicy,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    camera_keys_to_show: list[str] | None,
    device: torch.device,
) -> tuple[list[FeatureRecord], list[dict]]:
    vlm_expert = policy.model.vlm_with_expert
    vlm_model = vlm_expert.get_vlm_model()
    vision_model = vlm_model.vision_model
    connector = vlm_model.connector

    patch_size = getattr(vision_model.config, "patch_size", None)
    scale_factor = getattr(vlm_model.config, "scale_factor", None)

    records: list[FeatureRecord] = []
    metadata_rows: list[dict] = []

    with torch.no_grad():
        for frame_idx in frame_indices:
            frame = dataset[frame_idx]
            batch = prepare_frame_batch(policy, frame, device)
            if not batch:
                print(f"Skip frame {frame_idx}: no configured image keys found.")
                continue

            images, img_masks = policy.prepare_images(batch)
            ordered_keys = actual_image_order(policy, batch)
            if len(ordered_keys) != len(images):
                raise RuntimeError(
                    f"Image order mismatch: keys={len(ordered_keys)} images={len(images)}"
                )

            for camera_key, prepared_img, img_mask in zip(ordered_keys, images, img_masks, strict=False):
                if camera_keys_to_show is not None and camera_key not in camera_keys_to_show:
                    continue
                if camera_key not in frame:
                    continue

                original_rgb = chw_float01_to_rgb_uint8(frame[camera_key])
                model_input_rgb = model_input_to_rgb_uint8(prepared_img)

                pixel_values = prepared_img.to(device=device, dtype=vision_model.dtype)
                vision_outputs = vision_model(pixel_values=pixel_values, patch_attention_mask=None)
                vision_tokens_t = vision_outputs.last_hidden_state
                connector_tokens_t = connector(vision_tokens_t)

                # This is the visual token tensor appended by VLAFlowMatching.embed_prefix,
                # excluding optional non-spatial image special tokens.
                embed_dim = int(connector_tokens_t.shape[-1])
                final_visual_tokens_t = connector_tokens_t * math.sqrt(embed_dim)

                embed_image_tokens_t = vlm_expert.embed_image(prepared_img)
                diff = (embed_image_tokens_t - connector_tokens_t).abs().max().float().item()

                in_h, in_w = original_rgb.shape[:2]
                prep_h, prep_w = model_input_rgb.shape[:2]
                vision_grid = infer_vision_grid(
                    int(vision_tokens_t.shape[1]), (prep_h, prep_w), patch_size
                )
                connector_grid = infer_connector_grid(
                    int(connector_tokens_t.shape[1]), vision_grid, scale_factor
                )

                rec = FeatureRecord(
                    frame_idx=frame_idx,
                    camera_key=camera_key,
                    original_rgb=original_rgb,
                    model_input_rgb=model_input_rgb,
                    vision_tokens=vision_tokens_t[0].detach().float().cpu().numpy(),
                    connector_tokens=connector_tokens_t[0].detach().float().cpu().numpy(),
                    final_visual_tokens=final_visual_tokens_t[0].detach().float().cpu().numpy(),
                    vision_grid=vision_grid,
                    connector_grid=connector_grid,
                    input_shape=(in_h, in_w),
                    model_input_shape=(prep_h, prep_w),
                    connector_embed_dim=embed_dim,
                    embed_image_max_abs_diff=diff,
                )
                records.append(rec)
                metadata_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "camera_key": camera_key,
                        "dataset_image_hw": [in_h, in_w],
                        "model_input_hw": [prep_h, prep_w],
                        "vision_tokens": int(vision_tokens_t.shape[1]),
                        "vision_grid": list(vision_grid),
                        "connector_tokens": int(connector_tokens_t.shape[1]),
                        "connector_grid": list(connector_grid),
                        "connector_embed_dim": embed_dim,
                        "final_visual_tokens": int(final_visual_tokens_t.shape[1]),
                        "img_mask_true": bool(img_mask[0].item()) if img_mask.numel() else None,
                        "embed_image_max_abs_diff": diff,
                    }
                )
    return records, metadata_rows


def fit_pca_bundle(records: list[FeatureRecord], attr: str) -> tuple[SimplePCA, np.ndarray, np.ndarray, float]:
    tokens = np.concatenate([getattr(r, attr) for r in records], axis=0)
    pca = fit_simple_pca(tokens, n_components=3)
    transformed = pca.transform(tokens)
    lo = np.percentile(transformed, 1, axis=0)
    hi = np.percentile(transformed, 99, axis=0)
    hi = np.where(np.abs(hi - lo) < 1e-9, lo + 1.0, hi)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return pca, lo, hi, explained


def pca_rgb(
    tokens: np.ndarray,
    grid_hw: tuple[int, int],
    pca: SimplePCA,
    lo: np.ndarray,
    hi: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    h, w = grid_hw
    if h * w != tokens.shape[0]:
        raise ValueError(f"Grid {h}x{w} does not match {tokens.shape[0]} tokens")
    rgb = pca.transform(tokens)
    rgb = (rgb - lo) / (hi - lo)
    rgb = np.clip(rgb, 0.0, 1.0).reshape(h, w, 3)
    out_h, out_w = target_hw
    rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return (rgb * 255.0).round().clip(0, 255).astype(np.uint8)


def token_norm_rgb(
    tokens: np.ndarray,
    grid_hw: tuple[int, int],
    lo: float,
    hi: float,
    target_hw: tuple[int, int],
) -> np.ndarray:
    h, w = grid_hw
    if h * w != tokens.shape[0]:
        raise ValueError(f"Grid {h}x{w} does not match {tokens.shape[0]} tokens")
    norms = np.linalg.norm(tokens, axis=1).reshape(h, w)
    norm01 = np.clip((norms - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    out_h, out_w = target_hw
    norm01 = cv2.resize(norm01, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    heat_u8 = (norm01 * 255.0).round().clip(0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def resize_letterbox(img: np.ndarray, width: int, height: int, bg: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def wrap_label(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for chunk in text.replace("/", ".").split("."):
        candidate = chunk if not current else f"{current}.{chunk}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = chunk
    if current:
        parts.append(current)
    out: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            out.append(part)
        else:
            out.extend([part[i : i + max_chars] for i in range(0, len(part), max_chars)])
    return out


def draw_lines(
    img: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    *,
    font_scale: float = 0.42,
    color: tuple[int, int, int] = (20, 20, 20),
    line_height: int = 17,
) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


def make_tile(
    img: np.ndarray,
    lines: list[str],
    *,
    tile_w: int = 330,
    image_h: int = 260,
    title_h: int = 82,
) -> np.ndarray:
    tile = np.full((title_h + image_h, tile_w, 3), 255, dtype=np.uint8)
    body = resize_letterbox(img, tile_w, image_h, bg=245)
    tile[title_h:, :] = body
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrap_label(line, 38))
    draw_lines(tile, wrapped[:4], 8, 18)
    cv2.rectangle(tile, (0, 0), (tile_w - 1, title_h + image_h - 1), (220, 220, 220), 1)
    return tile


def make_banner(width: int, lines: list[str], height: int = 76) -> np.ndarray:
    banner = np.full((height, width, 3), 245, dtype=np.uint8)
    draw_lines(banner, lines, 12, 24, font_scale=0.52, line_height=21)
    return banner


def save_visualizations(records: list[FeatureRecord], output_dir: Path, max_rows_per_fig: int) -> dict:
    if max_rows_per_fig < 1:
        raise ValueError("--max_rows_per_fig must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_info = {
        "vision_tokens": fit_pca_bundle(records, "vision_tokens"),
        "connector_tokens": fit_pca_bundle(records, "connector_tokens"),
        "final_visual_tokens": fit_pca_bundle(records, "final_visual_tokens"),
    }
    final_norms = np.concatenate([np.linalg.norm(r.final_visual_tokens, axis=1) for r in records])
    norm_lo, norm_hi = np.percentile(final_norms, [1, 99])

    explained = {name: bundle[3] for name, bundle in pca_info.items()}
    explained["final_visual_token_norm_p01"] = float(norm_lo)
    explained["final_visual_token_norm_p99"] = float(norm_hi)
    chunks = [records[i : i + max_rows_per_fig] for i in range(0, len(records), max_rows_per_fig)]

    for chunk_idx, chunk in enumerate(chunks):
        row_images_for_chunk: list[np.ndarray] = []
        for row_idx, rec in enumerate(chunk):
            target_hw = rec.model_input_shape

            vision_rgb = pca_rgb(
                rec.vision_tokens,
                rec.vision_grid,
                pca_info["vision_tokens"][0],
                pca_info["vision_tokens"][1],
                pca_info["vision_tokens"][2],
                target_hw,
            )
            connector_rgb = pca_rgb(
                rec.connector_tokens,
                rec.connector_grid,
                pca_info["connector_tokens"][0],
                pca_info["connector_tokens"][1],
                pca_info["connector_tokens"][2],
                target_hw,
            )
            final_rgb = pca_rgb(
                rec.final_visual_tokens,
                rec.connector_grid,
                pca_info["final_visual_tokens"][0],
                pca_info["final_visual_tokens"][1],
                pca_info["final_visual_tokens"][2],
                target_hw,
            )
            final_norm_rgb = token_norm_rgb(
                rec.final_visual_tokens,
                rec.connector_grid,
                float(norm_lo),
                float(norm_hi),
                target_hw,
            )

            row_images = [
                rec.original_rgb,
                rec.model_input_rgb,
                vision_rgb,
                connector_rgb,
                final_rgb,
                final_norm_rgb,
            ]
            titles = [
                ["Original", f"frame={rec.frame_idx}", rec.camera_key],
                ["Actual model input", f"{rec.model_input_shape[0]}x{rec.model_input_shape[1]}"],
                ["Raw vision output PCA", f"{rec.vision_grid[0]}x{rec.vision_grid[1]} tokens"],
                ["Connector output PCA", f"{rec.connector_grid[0]}x{rec.connector_grid[1]} tokens"],
                ["Final visual tokens PCA", f"connector * sqrt({rec.connector_embed_dim})"],
                ["Final token norm", "same scale across rows"],
            ]
            tiles = [
                make_tile(img, title_lines)
                for img, title_lines in zip(row_images, titles, strict=True)
            ]
            row = np.hstack(tiles)
            if row_idx < len(chunk) - 1:
                gap = np.full((10, row.shape[1], 3), 235, dtype=np.uint8)
                row = np.vstack([row, gap])
            row_images_for_chunk.append(row)

        body = np.vstack(row_images_for_chunk)
        banner = make_banner(
            body.shape[1],
            [
                "Corrected SmolVLA visual feature PCA",
                "Uses SmolVLAPolicy.prepare_images() and the same vision_model -> connector path as inference.",
            ],
        )
        composed = np.vstack([banner, body])
        suffix = f"_part{chunk_idx:02d}" if len(chunks) > 1 else ""
        out_path = output_dir / f"smolvla_pca_corrected{suffix}.png"
        ok = cv2.imwrite(str(out_path), cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write {out_path}")

    return {k: float(v) for k, v in explained.items()}


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    policy: SmolVLAPolicy,
    pca_explained_variance: dict,
    rows: list[dict],
) -> None:
    metadata = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "frames": parse_frame_indices(args.frames),
        "camera_keys_requested": parse_camera_keys(args.camera_keys),
        "model_config_image_features": list(policy.config.image_features),
        "resize_imgs_with_padding": list(policy.config.resize_imgs_with_padding)
        if policy.config.resize_imgs_with_padding is not None
        else None,
        "add_image_special_tokens": bool(policy.model.add_image_special_tokens),
        "global_image_start_token_len": int(policy.model.global_image_start_token.numel()),
        "image_end_token_len": int(policy.model.image_end_token.numel()),
        "note": (
            "The final_visual_tokens visualization is connector output multiplied by sqrt(embed_dim), "
            "matching the spatial visual embeddings appended by embed_prefix. Optional image special "
            "tokens are non-spatial and are recorded here but not drawn as image grids."
        ),
        "pca_explained_variance_sum": pca_explained_variance,
        "records": rows,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Human-readable PCA visualization for SmolVLA visual features using the actual "
            "SmolVLA inference image path."
        )
    )
    parser.add_argument(
        "--dataset_id",
        default="wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge",
    )
    parser.add_argument(
        "--model_id",
        default="wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge0420-model",
    )
    parser.add_argument(
        "--frames",
        default="0,50,100,150,200,250,300,350,400",
        help="Comma list like 0,50,100 or range like 0:500:50.",
    )
    parser.add_argument(
        "--camera_keys",
        default=None,
        help="Optional comma-separated cameras to draw. Default draws all model image features present in the frame.",
    )
    parser.add_argument("--output_dir", default="smolvla_pca_corrected")
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_rows_per_fig", type=int, default=12)
    args = parser.parse_args()

    frame_indices = parse_frame_indices(args.frames)
    camera_keys_to_show = parse_camera_keys(args.camera_keys)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"Loading dataset: {args.dataset_id}")
    dataset = LeRobotDataset(args.dataset_id, video_backend=args.video_backend)

    print(f"Loading SmolVLA policy: {args.model_id}")
    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.to(device)
    policy.eval()

    if device.type == "cpu":
        vlm_model = policy.model.vlm_with_expert.get_vlm_model()
        vlm_model.vision_model.to(torch.float32)
        vlm_model.connector.to(torch.float32)

    print("Extracting visual features from the actual SmolVLA image path...")
    records, metadata_rows = extract_records(
        policy=policy,
        dataset=dataset,
        frame_indices=frame_indices,
        camera_keys_to_show=camera_keys_to_show,
        device=device,
    )
    if not records:
        raise RuntimeError("No records were extracted. Check --frames and --camera_keys.")

    print(f"Rendering {len(records)} rows to {output_dir}")
    pca_explained = save_visualizations(records, output_dir, args.max_rows_per_fig)
    save_metadata(output_dir, args, policy, pca_explained, metadata_rows)

    print(f"Done. Output directory: {output_dir}")
    print(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
