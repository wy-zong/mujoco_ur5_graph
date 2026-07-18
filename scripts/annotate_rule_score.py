#!/usr/bin/env python
"""
Generate rule-based per-frame scores for a LeRobot dataset and save them as a separate parquet file.

Expected dataset layout:
  <dataset_root>/data/chunk-*/file-*.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate rule_score parquet for LeRobot datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root folder of the dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to <dataset_root>/rule_scores.parquet",
    )
    parser.add_argument(
        "--intervention-key",
        type=str,
        default="complementary_info.is_intervention",
        help="Column key used to mark human intervention.",
    )
    parser.add_argument("--pre-window", type=int, default=8, help="Number of steps before intervention to penalize.")
    parser.add_argument("--score-low", type=float, default=-0.5, help="Score for pre-intervention steps.")
    parser.add_argument("--score-mid", type=float, default=1.0, help="Score for non-intervention steps.")
    parser.add_argument("--score-high", type=float, default=1.5, help="Score for intervention steps.")
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="Linear mapping scale for progress_sparse: clip(scale * score + bias, 0, max).",
    )
    parser.add_argument("--weight-bias", type=float, default=1.0, help="Linear mapping bias for progress_sparse.")
    parser.add_argument("--weight-max", type=float, default=3.0, help="Upper bound for progress_sparse.")
    return parser.parse_args()


def _to_bool(value) -> bool:
    if isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        value = value.reshape(-1)[0]
    return bool(value)


def load_index_frame_table(dataset_root: Path, intervention_key: str) -> pd.DataFrame:
    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    frames = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        base_cols = ["index", "episode_index"]
        missing = [c for c in base_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{file_path} missing required columns: {missing}")

        keep = df[base_cols].copy()
        if intervention_key in df.columns:
            keep[intervention_key] = df[intervention_key].map(_to_bool)
        else:
            keep[intervention_key] = False
        frames.append(keep)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("index").reset_index(drop=True)
    return out


def score_episode(
    intervention_flags: np.ndarray,
    score_low: float,
    score_mid: float,
    score_high: float,
    pre_window: int,
) -> np.ndarray:
    scores = np.full(intervention_flags.shape[0], score_mid, dtype=np.float32)
    intervention_indices = np.where(intervention_flags)[0]
    if intervention_indices.size == 0:
        return scores

    for idx in intervention_indices:
        start = max(0, idx - pre_window)
        if start < idx:
            scores[start:idx] = np.minimum(scores[start:idx], score_low)
        scores[idx] = score_high
    return scores


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_path = args.output.resolve() if args.output else (dataset_root / "rule_scores.parquet")

    df = load_index_frame_table(dataset_root, args.intervention_key)
    score_list = []

    for _, ep_df in df.groupby("episode_index", sort=False):
        flags = ep_df[args.intervention_key].to_numpy(dtype=bool)
        ep_scores = score_episode(
            intervention_flags=flags,
            score_low=args.score_low,
            score_mid=args.score_mid,
            score_high=args.score_high,
            pre_window=args.pre_window,
        )
        score_list.append(pd.Series(ep_scores, index=ep_df.index))

    df["rule_score"] = pd.concat(score_list).sort_index().astype(np.float32)
    mapped = args.weight_scale * df["rule_score"].to_numpy() + args.weight_bias
    df["progress_sparse"] = np.clip(mapped, 0.0, args.weight_max).astype(np.float32)

    out_df = df[["index", "episode_index", "rule_score", "progress_sparse"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"Saved rule score annotations: {output_path}")
    print(f"Rows: {len(out_df)}")
    print(
        "Score stats:",
        f"min={out_df['rule_score'].min():.3f}",
        f"mean={out_df['rule_score'].mean():.3f}",
        f"max={out_df['rule_score'].max():.3f}",
    )


if __name__ == "__main__":
    main()
