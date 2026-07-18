from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


def _disable_broken_torchvision_detection() -> None:
    """Some local envs expose only torchvision metadata/DLLs, which breaks transformers imports."""
    try:
        import torchvision  # type: ignore

        if hasattr(torchvision, "io"):
            return

        import transformers.utils.import_utils as transformers_import_utils

        transformers_import_utils._torchvision_available = False  # noqa: SLF001
        transformers_import_utils._torchvision_version = "N/A"  # noqa: SLF001
    except Exception:
        pass


_disable_broken_torchvision_detection()


DEFAULT_DATASET_ID = "wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge"
DEFAULT_MODEL_ID = "lerobot/pi05_libero_finetuned"
RESIZE_WITH_PAD_TORCH = None


class PCA3:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "PCA3":
        values = values.astype(np.float32, copy=False)
        self.mean_ = values.mean(axis=0, keepdims=True)
        centered = values - self.mean_
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vh[:3].T.astype(np.float32, copy=False)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA3 must be fit before transform")
        values = values.astype(np.float32, copy=False)
        return (values - self.mean_) @ self.components_


def parse_frame_indices(value: str) -> list[int]:
    frames = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        frames.append(int(part))
    if not frames:
        raise argparse.ArgumentTypeError("frames must contain at least one integer")
    return frames


def parse_camera_keys(values: list[str] | None) -> list[str]:
    if not values:
        return ["observation.images.left_camera1"]

    keys: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                keys.append(part)
    return keys


def tensor_image_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    img = img_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    img_np = (img.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def model_input_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert [1, C, H, W] in [-1, 1] back to uint8 RGB for inspection."""
    img = img_tensor.detach().cpu()[0]
    img = ((img + 1.0) / 2.0).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def preprocess_for_pi05(
    img_tensor: torch.Tensor,
    image_resolution: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Mirror PI05Policy._preprocess_images for one image key."""
    img = img_tensor.detach()
    if img.ndim == 3:
        img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    is_channels_first = img.shape[1] == 3
    if is_channels_first:
        img = img.permute(0, 2, 3, 1)

    if tuple(img.shape[1:3]) != tuple(image_resolution):
        if RESIZE_WITH_PAD_TORCH is None:
            raise RuntimeError("resize_with_pad_torch is not loaded")
        img = RESIZE_WITH_PAD_TORCH(img, *image_resolution)

    img = img * 2.0 - 1.0

    if is_channels_first:
        img = img.permute(0, 3, 1, 2)

    return img


def infer_spatial_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], str]:
    """Return tokens that can be reshaped to a 2D patch grid."""
    if tokens.ndim == 3:
        tokens = tokens[0]

    n_tokens = int(tokens.shape[0])
    side = int(math.isqrt(n_tokens))
    if side * side == n_tokens:
        return tokens, (side, side), ""

    no_cls = n_tokens - 1
    side = int(math.isqrt(no_cls))
    if side * side == no_cls:
        return tokens[1:], (side, side), "dropped first token"

    factors = [(h, n_tokens // h) for h in range(1, int(math.sqrt(n_tokens)) + 1) if n_tokens % h == 0]
    if factors:
        grid_h, grid_w = min(factors, key=lambda hw: abs(hw[0] - hw[1]))
        return tokens, (grid_h, grid_w), f"inferred non-square grid {grid_h}x{grid_w}"

    raise ValueError(f"Cannot infer a spatial grid from {n_tokens} tokens")


def extract_final_image_features(paligemma_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Use the same final image feature path as PI05 embed_image."""
    image_outputs = paligemma_model.get_image_features(pixel_values)

    if isinstance(image_outputs, torch.Tensor):
        features = image_outputs
    elif getattr(image_outputs, "pooler_output", None) is not None:
        features = image_outputs.pooler_output
    elif getattr(image_outputs, "last_hidden_state", None) is not None:
        features = image_outputs.last_hidden_state
    else:
        raise TypeError(f"Unsupported get_image_features output type: {type(image_outputs)!r}")

    hidden_size = paligemma_model.config.text_config.hidden_size
    return features * (hidden_size**0.5)


def fit_stage_pca(samples: list[dict], token_key: str) -> PCA3:
    all_tokens = np.concatenate([sample[token_key] for sample in samples], axis=0)
    return PCA3().fit(all_tokens)


def tokens_to_pca_image(tokens: np.ndarray, pca: PCA3, grid_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    features = pca.transform(tokens)
    min_v = features.min(axis=0, keepdims=True)
    max_v = features.max(axis=0, keepdims=True)
    features = (features - min_v) / (max_v - min_v + 1e-8)
    pca_image = features.reshape(grid_hw[0], grid_hw[1], 3)
    return cv2.resize(pca_image, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


def array_to_pil(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def fit_to_tile(image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    tile_w, tile_h = tile_size
    image = image.convert("RGB")
    image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_w, tile_h), "white")
    x = (tile_w - image.width) // 2
    y = (tile_h - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str = "black") -> None:
    x, y = xy
    for line in text.split("\n"):
        draw.text((x, y), line, fill=fill)
        y += 14


def save_contact_sheet(
    samples: list[dict],
    args: argparse.Namespace,
    image_resolution: tuple[int, int],
    output_path: Path,
    vision_pca: PCA3,
    final_pca: PCA3,
) -> None:
    tile_size = (320, 240)
    header_h = 42
    pad = 10
    title_h = 82
    cols = 4
    rows = len(samples)

    width = cols * tile_size[0] + (cols + 1) * pad
    row_h = header_h + tile_size[1] + pad
    height = title_h + rows * row_h + pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 12), "PI05 Vision PCA", fill="black")
    draw.text((pad, 30), f"model={args.model_id}", fill="black")
    adapter = getattr(args, "resolved_lora_adapter", None)
    if adapter:
        draw.text((pad, 46), f"lora_adapter={adapter}", fill="black")
        draw.text((pad, 62), f"dataset={args.dataset_id}", fill="black")
    else:
        draw.text((pad, 46), f"dataset={args.dataset_id}", fill="black")

    for row_idx, sample in enumerate(samples):
        y0 = title_h + row_idx * row_h
        orig = sample["original"]
        model_input = sample["model_input"]
        target_hw = orig.shape[:2]
        vision_img = tokens_to_pca_image(
            sample["vision_tokens"], vision_pca, sample["vision_grid"], target_hw
        )
        final_img = tokens_to_pca_image(
            sample["final_tokens"], final_pca, sample["final_grid"], target_hw
        )

        cells = [
            (
                f"Original\nframe {sample['frame']} | {sample['camera']}",
                array_to_pil(orig),
            ),
            (
                f"Actual PI05 input\n{image_resolution[0]}x{image_resolution[1]} resize+pad",
                array_to_pil(model_input),
            ),
            (
                f"Vision tower output PCA\n{sample['vision_grid'][0]}x{sample['vision_grid'][1]} patches",
                array_to_pil(vision_img),
            ),
            (
                f"Final image embedding PCA\n{sample['final_grid'][0]}x{sample['final_grid'][1]} tokens",
                array_to_pil(final_img),
            ),
        ]

        for col_idx, (caption, image) in enumerate(cells):
            x0 = pad + col_idx * (tile_size[0] + pad)
            draw.rectangle((x0, y0, x0 + tile_size[0], y0 + header_h + tile_size[1]), outline="#dddddd")
            draw_multiline(draw, (x0 + 6, y0 + 6), caption)
            canvas.paste(fit_to_tile(image, tile_size), (x0, y0 + header_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def has_local_adapter_config(model_id: str) -> bool:
    return (Path(model_id) / "adapter_config.json").is_file()


def load_pi05_base_policy(model_id: str, local_files_only: bool, strict: bool):
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    print(f"正在載入 PI05 模型: {model_id}")
    return PI05Policy.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        strict=strict,
    )


def load_policy(
    model_id: str,
    device: str,
    local_files_only: bool,
    strict: bool,
    lora_adapter: str | None = None,
    base_model_id: str | None = None,
):
    adapter_id = lora_adapter
    base_id = base_model_id
    model_id_is_adapter = False

    if adapter_id is None and has_local_adapter_config(model_id):
        adapter_id = model_id
        model_id_is_adapter = True
    elif adapter_id is None:
        base_id = base_id or model_id

    if adapter_id is not None:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "載入 LoRA adapter 需要 peft 套件。請在你用 LeRobot LoRA 微調的同一個環境執行，"
                "或安裝 peft 後再跑。"
            ) from exc

        print(f"正在載入 LoRA adapter: {adapter_id}")
        peft_config = PeftConfig.from_pretrained(adapter_id, local_files_only=local_files_only)
        if not base_id:
            base_id = peft_config.base_model_name_or_path
        if not base_id and not model_id_is_adapter:
            base_id = model_id
        if not base_id:
            raise ValueError(
                "LoRA adapter_config.json 沒有 base_model_name_or_path。請用 --base_model_id 指定 base pi05 模型。"
            )
        print(f"LoRA base PI05 模型: {base_id}")
        policy = load_pi05_base_policy(base_id, local_files_only=local_files_only, strict=strict)
        policy = PeftModel.from_pretrained(
            policy,
            adapter_id,
            config=peft_config,
            local_files_only=local_files_only,
            is_trainable=False,
        )
    else:
        policy = load_pi05_base_policy(model_id, local_files_only=local_files_only, strict=strict)

    actual_device = torch.device(device)
    policy.to(actual_device)
    policy.eval()
    return policy


def unwrap_pi05_policy(policy):
    def is_pi05_policy(obj) -> bool:
        model = getattr(obj, "model", None)
        return model is not None and hasattr(model, "paligemma_with_expert")

    seen: set[int] = set()
    candidates = [policy]
    while candidates:
        obj = candidates.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if is_pi05_policy(obj):
            return obj

        get_base_model = getattr(obj, "get_base_model", None)
        if callable(get_base_model):
            try:
                candidates.append(get_base_model())
            except Exception:
                pass

        for attr in ("base_model", "model", "module"):
            child = getattr(obj, attr, None)
            if child is not None:
                candidates.append(child)

    raise TypeError("無法從載入的模型中找到 PI05Policy。請確認 base model / LoRA adapter 是 pi05。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize PI05 vision tower output and final image embeddings with PCA."
    )
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID, help="LeRobot dataset repo id/path")
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "PI05 pretrained model repo id/path. If this local dir contains adapter_config.json, it is treated "
            "as a LeRobot LoRA adapter dir."
        ),
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="Optional LeRobot/PEFT LoRA adapter repo id/path, usually checkpoints/.../pretrained_model.",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default=None,
        help="Override the LoRA adapter's base_model_name_or_path when it is missing or points to a moved path.",
    )
    parser.add_argument(
        "--camera_key",
        action="append",
        default=None,
        help="Camera key to visualize. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--frames",
        type=parse_frame_indices,
        default=parse_frame_indices("0,50,100,150,200,250,300,350,400"),
        help="Comma-separated dataset frame indices.",
    )
    parser.add_argument("--output", type=str, default="pi05_pca_visualization.png", help="Output PNG path")
    parser.add_argument("--video_backend", type=str, default="pyav", help="LeRobot video backend")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model inference.",
    )
    parser.add_argument("--local_files_only", action="store_true", help="Do not download model/dataset files")
    parser.add_argument(
        "--non_strict",
        action="store_true",
        help="Load PI05 checkpoint with strict=False if the checkpoint has small key mismatches.",
    )
    args = parser.parse_args()

    camera_keys = parse_camera_keys(args.camera_key)
    output_path = Path(args.output)
    args.resolved_lora_adapter = args.lora_adapter
    if args.resolved_lora_adapter is None and has_local_adapter_config(args.model_id):
        args.resolved_lora_adapter = args.model_id

    global RESIZE_WITH_PAD_TORCH  # noqa: PLW0603
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
    except Exception as exc:
        raise RuntimeError(
            "無法匯入 LeRobot/pi05 依賴。請在可正常執行 lerobot 的 Python/uv/conda 環境中跑此腳本。"
        ) from exc
    RESIZE_WITH_PAD_TORCH = resize_with_pad_torch

    print(f"正在載入資料集: {args.dataset_id}")
    dataset = LeRobotDataset(args.dataset_id, video_backend=args.video_backend)
    policy = load_policy(
        args.model_id,
        args.device,
        args.local_files_only,
        strict=not args.non_strict,
        lora_adapter=args.lora_adapter,
        base_model_id=args.base_model_id,
    )
    pi05_policy = unwrap_pi05_policy(policy)

    device = next(pi05_policy.parameters()).device
    image_resolution = tuple(pi05_policy.config.image_resolution)
    paligemma_model = pi05_policy.model.paligemma_with_expert.paligemma.model
    vision_tower = paligemma_model.vision_tower

    samples: list[dict] = []
    print("正在提取 PI05 視覺特徵...")
    with torch.inference_mode():
        for frame_idx in args.frames:
            try:
                item = dataset[frame_idx]
            except IndexError:
                print(f"警告: frame {frame_idx} 超出資料集長度，略過。")
                continue

            for camera_key in camera_keys:
                if camera_key not in item:
                    print(f"警告: frame {frame_idx} 找不到 camera key: {camera_key}，略過。")
                    continue

                original_pil = tensor_image_to_pil(item[camera_key])
                pixel_values = preprocess_for_pi05(item[camera_key], image_resolution, device)

                vision_outputs = vision_tower(pixel_values=pixel_values.to(torch.float32))
                raw_tokens, raw_grid, raw_note = infer_spatial_tokens(vision_outputs.last_hidden_state)

                final_features = extract_final_image_features(paligemma_model, pixel_values)
                final_tokens, final_grid, final_note = infer_spatial_tokens(final_features)

                samples.append(
                    {
                        "frame": frame_idx,
                        "camera": camera_key,
                        "original": np.asarray(original_pil),
                        "model_input": model_input_to_rgb(pixel_values),
                        "vision_tokens": raw_tokens.detach().float().cpu().numpy(),
                        "vision_grid": raw_grid,
                        "vision_note": raw_note,
                        "final_tokens": final_tokens.detach().float().cpu().numpy(),
                        "final_grid": final_grid,
                        "final_note": final_note,
                    }
                )

    if not samples:
        raise RuntimeError("沒有取得任何影像樣本，請檢查 --dataset_id / --camera_key / --frames")

    vision_pca = fit_stage_pca(samples, "vision_tokens")
    final_pca = fit_stage_pca(samples, "final_tokens")

    save_contact_sheet(samples, args, image_resolution, output_path, vision_pca, final_pca)
    print(f"完成: {output_path.resolve()}")

    notes = sorted(
        {
            note
            for sample in samples
            for note in (sample["vision_note"], sample["final_note"])
            if note
        }
    )
    if notes:
        print("Grid notes: " + "; ".join(notes))


if __name__ == "__main__":
    main()
