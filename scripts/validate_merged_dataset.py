#!/usr/bin/env python
"""Validate that a merged LeRobot dataset correctly contains two source datasets.

Usage:
python scripts/validate_merged_dataset.py \
  --base-dir "C:/Users/ccu/mujoco_ur5_graph/outputs" \
  --src1 "full-fold-the-rag-parquet" \
  --src2 "full-fold-the-rag-parquet-c" \
  --merged "full-fold-the-rag-parquet-merged" \
  --sample-rows 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))


def is_nonempty_image_cell(value) -> bool:
    if isinstance(value, dict):
        b = value.get("bytes")
        p = value.get("path")
        if isinstance(b, (bytes, bytearray)):
            return len(b) > 0
        if b is not None:
            return True
        return isinstance(p, str) and len(p) > 0
    if isinstance(value, (bytes, bytearray)):
        return len(value) > 0
    if isinstance(value, str):
        return len(value) > 0
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--src1", type=str, required=True)
    parser.add_argument("--src2", type=str, required=True)
    parser.add_argument("--merged", type=str, required=True)
    parser.add_argument("--sample-rows", type=int, default=20)
    args = parser.parse_args()

    src1_dir = args.base_dir / args.src1
    src2_dir = args.base_dir / args.src2
    merged_dir = args.base_dir / args.merged

    i1 = load_info(src1_dir)
    i2 = load_info(src2_dir)
    im = load_info(merged_dir)

    expected_eps = i1["total_episodes"] + i2["total_episodes"]
    expected_frames = i1["total_frames"] + i2["total_frames"]

    parquet_files = sorted((merged_dir / "data").rglob("*.parquet"))
    row_count = 0
    episode_values: set[int] = set()
    cam1_nonempty = 0
    cam3_nonempty = 0
    sample_checked = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        row_count += len(df)

        if "episode_index" in df.columns:
            episode_values.update(df["episode_index"].dropna().astype(int).tolist())

        idxs = list(range(len(df)))
        random.shuffle(idxs)
        for idx in idxs[: min(5, len(df))]:
            if sample_checked >= args.sample_rows:
                break
            row = df.iloc[idx]

            if is_nonempty_image_cell(row.get("observation.images.camera1", None)):
                cam1_nonempty += 1
            if is_nonempty_image_cell(row.get("observation.images.camera3", None)):
                cam3_nonempty += 1
            sample_checked += 1

        if sample_checked >= args.sample_rows:
            break

    min_ep = min(episode_values) if episode_values else None
    max_ep = max(episode_values) if episode_values else None
    expected_ep_set = set(range(im["total_episodes"]))
    continuous = episode_values == expected_ep_set

    has_cam1 = "observation.images.camera1" in im.get("features", {})
    has_cam3 = "observation.images.camera3" in im.get("features", {})
    cam1_dtype = im.get("features", {}).get("observation.images.camera1", {}).get("dtype")
    cam3_dtype = im.get("features", {}).get("observation.images.camera3", {}).get("dtype")

    print("=== MERGE VALIDATION ===")
    print(f"src episodes: {i1['total_episodes']} + {i2['total_episodes']} = {expected_eps}")
    print(f"merged episodes: {im['total_episodes']}")
    print(f"src frames: {i1['total_frames']} + {i2['total_frames']} = {expected_frames}")
    print(f"merged frames(meta): {im['total_frames']}")
    print(f"merged frames(parquet rows): {row_count}")
    print(f"fps src/merged: {i1['fps']} / {i2['fps']} / {im['fps']}")
    print(f"episode_index min/max: {min_ep}/{max_ep}")
    print(f"episode_index continuous 0..N-1: {continuous}")
    print(f"sample checked rows: {sample_checked}")
    print(f"camera1 non-empty samples: {cam1_nonempty}/{sample_checked}")
    print(f"camera3 non-empty samples: {cam3_nonempty}/{sample_checked}")
    print(f"has camera1 feature: {has_cam1}")
    print(f"has camera3 feature: {has_cam3}")
    print(f"camera1 dtype: {cam1_dtype}")
    print(f"camera3 dtype: {cam3_dtype}")

    overall_ok = (
        im["total_episodes"] == expected_eps
        and im["total_frames"] == expected_frames
        and row_count == im["total_frames"]
        and i1["fps"] == i2["fps"] == im["fps"]
        and continuous
        and has_cam1
        and has_cam3
        and cam1_dtype == "image"
        and cam3_dtype == "image"
        and sample_checked > 0
        and cam1_nonempty == sample_checked
        and cam3_nonempty == sample_checked
    )

    print(f"OVERALL_OK={overall_ok}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
