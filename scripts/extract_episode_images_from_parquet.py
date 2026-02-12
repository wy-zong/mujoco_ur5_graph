#!/usr/bin/env python
"""
從 LeRobot parquet 資料集中，提取指定 episode 的影像並重建為圖片檔。

用途：
- 用來檢查影像壓縮後的可接受程度（例如 JPEG 有損壓縮）。
- 只輸出到外部資料夾，不會修改原始資料集。

基本用法（PowerShell）：
    python scripts\\extract_episode_images_from_parquet.py `
      --dataset-root "C:\\Users\\ccu\\mujoco_ur5_graph\\outputs\\full-fold-the-rag-parquet" `
      --episode 0 `
      --output-dir "C:\\Users\\ccu\\mujoco_ur5_graph\\outputs\\inspect_reconstructed"

只抽前 200 張（快速檢查）：
    python scripts\\extract_episode_images_from_parquet.py `
      --dataset-root "C:\\Users\\ccu\\mujoco_ur5_graph\\outputs\\full-fold-the-rag-parquet" `
      --episode 0 `
      --output-dir "C:\\Users\\ccu\\mujoco_ur5_graph\\outputs\\inspect_reconstructed" `
      --max-frames 200
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image frames for a specific episode from a LeRobot parquet dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset root.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to extract.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where extracted images will be written.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on extracted frames for quick inspection.",
    )
    return parser.parse_args()


def get_image_columns(dataset_root: Path) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    image_cols = [
        key for key, spec in info.get("features", {}).items() if isinstance(spec, dict) and spec.get("dtype") == "image"
    ]
    if not image_cols:
        raise ValueError(f"No image columns found in {info_path}")
    return sorted(image_cols)


def iter_episode_rows(dataset_root: Path, episode_index: int, columns: list[str]):
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_ds = ds.dataset(str(data_dir), format="parquet")
    needed_cols = ["episode_index", "frame_index", *columns]
    filter_expr = pc.field("episode_index") == episode_index
    scanner = parquet_ds.scanner(columns=needed_cols, filter=filter_expr)

    for record_batch in scanner.to_batches():
        data = record_batch.to_pydict()
        n_rows = len(data["episode_index"])
        for i in range(n_rows):
            row = {key: data[key][i] for key in needed_cols}
            yield row


def save_image_cell(cell_value: dict, destination: Path, dataset_root: Path) -> None:
    if not isinstance(cell_value, dict):
        raise ValueError(f"Unexpected image cell format: {type(cell_value)}")

    image_bytes = cell_value.get("bytes")
    image_path = cell_value.get("path")

    if image_bytes is not None:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image_format = (img.format or "PNG").lower()
            target = destination.with_suffix(f".{image_format}")
            target.parent.mkdir(parents=True, exist_ok=True)
            img.save(target)
        return

    if image_path is None:
        raise ValueError("Image cell has neither bytes nor path.")

    source = Path(image_path)
    if not source.is_absolute():
        source = dataset_root / source
    if not source.exists():
        raise FileNotFoundError(f"Referenced image path does not exist: {source}")

    target = destination.with_suffix(source.suffix if source.suffix else ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_columns = get_image_columns(dataset_root)
    episode_dir = output_root / f"episode-{args.episode:06d}"
    count = 0

    for row in iter_episode_rows(dataset_root, args.episode, image_columns):
        frame_index = int(row["frame_index"])
        for col in image_columns:
            cam_dir = episode_dir / col
            dest = cam_dir / f"frame-{frame_index:06d}"
            save_image_cell(row[col], dest, dataset_root)

        count += 1
        if args.max_frames is not None and count >= args.max_frames:
            break

    if count == 0:
        raise ValueError(f"No rows found for episode {args.episode}")

    print(f"Extracted {count} frame(s) for episode {args.episode}")
    print(f"Output directory: {episode_dir}")


if __name__ == "__main__":
    main()
