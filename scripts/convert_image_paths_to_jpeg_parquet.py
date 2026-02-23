#!/usr/bin/env python
"""Convert a LeRobot image dataset to JPEG-bytes-in-parquet with per-episode parquet split.

Target format:
- image cells stored in parquet as {"bytes": <jpeg bytes>, "path": "frame-XXXXXX.jpg"}
- no images/ directory in output dataset
- one parquet file per episode: data/chunk-000/file-{episode_index:03d}.parquet
- meta/episodes updated so data/file_index == episode_index
"""


from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_episodes(dataset_dir: Path) -> pd.DataFrame:
    ep_files = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No meta episodes parquet found under: {dataset_dir / 'meta' / 'episodes'}")
    frames = [pd.read_parquet(p) for p in ep_files]
    df = pd.concat(frames, ignore_index=True)
    if "episode_index" not in df.columns:
        raise KeyError("meta/episodes is missing required column 'episode_index'")
    df = df.sort_values("episode_index").reset_index(drop=True)
    return df


def write_episodes(dataset_dir: Path, episodes_df: pd.DataFrame) -> None:
    ep_dir = dataset_dir / "meta" / "episodes"
    if ep_dir.exists():
        shutil.rmtree(ep_dir)
    out = ep_dir / "chunk-000" / "file-000.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(out, index=False)


def get_image_columns(info: dict) -> list[str]:
    features = info.get("features", {})
    cols = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "image"]
    return sorted(cols)


def normalize_output_image_name(frame_index_value, fallback_row_idx: int) -> str:
    try:
        n = int(frame_index_value)
    except Exception:
        n = fallback_row_idx
    return f"frame-{n:06d}.jpg"


def encode_jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def open_source_image(src_root: Path, cell_value):
    if isinstance(cell_value, dict):
        b = cell_value.get("bytes")
        p = cell_value.get("path")

        if isinstance(b, (bytes, bytearray)) and len(b) > 0:
            return Image.open(io.BytesIO(b))

        if isinstance(p, str) and p:
            source_path = src_root / p
            if source_path.exists():
                return Image.open(source_path)
            raise FileNotFoundError(f"Image path not found: {source_path}")

        raise ValueError("Image cell dict has neither valid 'bytes' nor valid 'path'.")

    if isinstance(cell_value, (bytes, bytearray)) and len(cell_value) > 0:
        return Image.open(io.BytesIO(cell_value))

    raise ValueError(f"Unsupported image cell type: {type(cell_value)}")


def convert_image_cells_in_df(df: pd.DataFrame, image_columns: list[str], src_root: Path, jpeg_quality: int) -> int:
    jpeg_bytes_total = 0
    for col in image_columns:
        if col not in df.columns:
            raise KeyError(f"Missing image column '{col}' in parquet rows")

    for row_idx in range(len(df)):
        frame_index_value = df.iloc[row_idx]["frame_index"] if "frame_index" in df.columns else row_idx
        out_name = normalize_output_image_name(frame_index_value, row_idx)

        for col in image_columns:
            cell = df.iloc[row_idx][col]
            with open_source_image(src_root, cell) as img:
                rgb = img.convert("RGB")
                encoded = encode_jpeg_bytes(rgb, jpeg_quality)
            jpeg_bytes_total += len(encoded)
            df.at[row_idx, col] = {"bytes": encoded, "path": out_name}

    return jpeg_bytes_total


def copy_meta(src_dir: Path, dst_dir: Path) -> None:
    src_meta = src_dir / "meta"
    dst_meta = dst_dir / "meta"
    if not src_meta.exists():
        raise FileNotFoundError(f"Missing meta directory: {src_meta}")
    shutil.copytree(src_meta, dst_meta)


def update_info_file(dst_dataset_dir: Path, total_episodes: int, total_frames: int) -> None:
    info_path = dst_dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["video_path"] = None
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding="utf-8")


def verify_target_format(dataset_dir: Path, sample_rows: int = 20) -> tuple[bool, str]:
    info = load_info(dataset_dir)
    image_columns = get_image_columns(info)

    if not image_columns:
        return False, "No image feature columns found in meta/info.json"

    if info.get("video_path", "__MISSING__") is not None:
        return False, "Expected video_path to be null for image dataset"

    if (dataset_dir / "images").exists():
        return False, "images/ directory exists; expected no images directory"

    parquet_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not parquet_files:
        return False, "No parquet files found in data/"

    # 1) verify per-episode file split rule from meta/episodes
    ep_df = load_episodes(dataset_dir)
    needed_ep_cols = {"episode_index", "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"}
    if not needed_ep_cols.issubset(set(ep_df.columns)):
        return False, f"meta/episodes missing required columns: {sorted(needed_ep_cols - set(ep_df.columns))}"

    expected_cursor = 0
    for _, row in ep_df.iterrows():
        ep = int(row["episode_index"])
        chk = int(row["data/chunk_index"])
        fidx = int(row["data/file_index"])
        frm = int(row["dataset_from_index"])
        to = int(row["dataset_to_index"])

        if chk != 0:
            return False, f"episode {ep} has data/chunk_index={chk}, expected 0"
        if fidx != ep:
            return False, f"episode {ep} has data/file_index={fidx}, expected episode_index"
        if frm != expected_cursor:
            return False, f"episode {ep} has dataset_from_index={frm}, expected {expected_cursor}"
        if to <= frm:
            return False, f"episode {ep} has invalid dataset_to_index={to} <= dataset_from_index={frm}"
        expected_cursor = to

    if expected_cursor != int(info.get("total_frames", -1)):
        return False, f"total_frames mismatch: info={info.get('total_frames')} vs episodes_end={expected_cursor}"

    # 2) verify each parquet only contains one matching episode
    file_map = {int(r["episode_index"]): (int(r["data/chunk_index"]), int(r["data/file_index"])) for _, r in ep_df.iterrows()}
    for ep, (chk, fidx) in file_map.items():
        p = dataset_dir / "data" / f"chunk-{chk:03d}" / f"file-{fidx:03d}.parquet"
        if not p.exists():
            return False, f"Missing expected parquet file for episode {ep}: {p}"
        scalar_df = pd.read_parquet(p, columns=["episode_index"])
        uniq = sorted(set(int(x) for x in scalar_df["episode_index"].tolist()))
        if uniq != [ep]:
            return False, f"Parquet {p.name} episode_index mismatch: {uniq}, expected [{ep}]"

    # 3) sample image cell validity using iter_batches
    checked = 0
    for p in parquet_files:
        pf = pq.ParquetFile(p)
        schema_names = set(pf.schema_arrow.names)
        for col in image_columns:
            if col not in schema_names:
                return False, f"Missing image column {col} in {p}"

        for batch in pf.iter_batches(batch_size=5, columns=image_columns):
            for row_idx in range(batch.num_rows):
                for col_idx, col in enumerate(image_columns):
                    cell = batch.column(col_idx)[row_idx].as_py()
                    if not isinstance(cell, dict):
                        return False, f"Image cell is not dict in {p} row {row_idx} col {col}"
                    b = cell.get("bytes")
                    img_path = cell.get("path")
                    if not isinstance(b, (bytes, bytearray)) or len(b) == 0:
                        return False, f"Empty/non-bytes image bytes in {p} row {row_idx} col {col}"
                    if not isinstance(img_path, str) or not img_path.lower().endswith(".jpg"):
                        return False, f"Image path is not .jpg in {p} row {row_idx} col {col}: {img_path}"
                checked += 1
                if checked >= sample_rows:
                    return True, f"OK (sampled {checked} rows)"

    return checked > 0, f"OK (sampled {checked} rows)" if checked > 0 else "No rows sampled"


def run_conversion(src_dataset_dir: Path, dst_dataset_dir: Path, jpeg_quality: int, overwrite: bool) -> None:
    if jpeg_quality < 1 or jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    if not src_dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset dir not found: {src_dataset_dir}")

    info = load_info(src_dataset_dir)
    image_columns = get_image_columns(info)

    if info.get("video_path", "__MISSING__") is not None:
        raise ValueError("Source dataset appears to be video mode (video_path is not null).")

    if not image_columns:
        raise ValueError("Source dataset has no image features to convert.")

    episodes_df = load_episodes(src_dataset_dir)

    if dst_dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_dataset_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_dataset_dir)

    dst_dataset_dir.mkdir(parents=True, exist_ok=True)
    copy_meta(src_dataset_dir, dst_dataset_dir)

    total_rows = 0
    total_jpeg_bytes = 0
    cursor = 0

    for _, row in episodes_df.iterrows():
        ep = int(row["episode_index"])
        src_chunk = int(row["data/chunk_index"])
        src_file = int(row["data/file_index"])
        src_parquet = src_dataset_dir / "data" / f"chunk-{src_chunk:03d}" / f"file-{src_file:03d}.parquet"

        if not src_parquet.exists():
            raise FileNotFoundError(f"Source parquet not found for episode {ep}: {src_parquet}")

        # Filter by episode to produce one-file-per-episode output.
        ep_df = pd.read_parquet(src_parquet, filters=[("episode_index", "==", ep)])
        if ep_df.empty:
            raise ValueError(f"No rows loaded for episode {ep} from {src_parquet}")

        jpeg_bytes = convert_image_cells_in_df(ep_df, image_columns=image_columns, src_root=src_dataset_dir, jpeg_quality=jpeg_quality)

        dst_parquet = dst_dataset_dir / "data" / "chunk-000" / f"file-{ep:03d}.parquet"
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(dst_parquet, index=False)

        rows = len(ep_df)
        total_rows += rows
        total_jpeg_bytes += jpeg_bytes

        episodes_df.loc[episodes_df["episode_index"] == ep, "data/chunk_index"] = 0
        episodes_df.loc[episodes_df["episode_index"] == ep, "data/file_index"] = ep
        episodes_df.loc[episodes_df["episode_index"] == ep, "dataset_from_index"] = cursor
        episodes_df.loc[episodes_df["episode_index"] == ep, "dataset_to_index"] = cursor + rows
        cursor += rows

        print(f"Converted episode {ep}: rows={rows}, jpeg_bytes={jpeg_bytes}, out={dst_parquet.name}")

    write_episodes(dst_dataset_dir, episodes_df)
    update_info_file(dst_dataset_dir, total_episodes=len(episodes_df), total_frames=cursor)

    ok, msg = verify_target_format(dst_dataset_dir, sample_rows=20)
    print("Verification:", msg)
    if not ok:
        raise RuntimeError("Converted dataset format verification failed")

    print("Done")
    print(f"Total rows converted: {total_rows}")
    print(f"Total JPEG bytes: {total_jpeg_bytes}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert path-based image dataset to JPEG-bytes-in-parquet format.")
    parser.add_argument("--src-dataset-dir", type=Path, required=True)
    parser.add_argument("--dst-dataset-dir", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        ok, msg = verify_target_format(args.src_dataset_dir, sample_rows=20)
        print(msg)
        return 0 if ok else 1

    if args.dst_dataset_dir is None:
        raise ValueError("--dst-dataset-dir is required unless --verify-only is used")

    run_conversion(
        src_dataset_dir=args.src_dataset_dir,
        dst_dataset_dir=args.dst_dataset_dir,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
