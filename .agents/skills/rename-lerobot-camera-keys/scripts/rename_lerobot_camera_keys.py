#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_MAPPING = {
    "observation.images.left_camera1": "observation.images.camera1",
    "observation.images.left_camera3": "observation.images.camera3",
}
VIDEO_META_SUFFIXES = ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]


def _rename_json_keys(path: Path, mapping: dict[str, str], top_key: str | None = None) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if top_key is None:
        obj = {mapping.get(k, k): v for k, v in obj.items()}
    else:
        container = obj.get(top_key, {})
        obj[top_key] = {mapping.get(k, k): v for k, v in container.items()}
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=4), encoding="utf-8")


def _rename_data_parquet_columns(dst: Path, mapping: dict[str, str]) -> None:
    for p in sorted((dst / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(p)
        cols = {c: mapping[c] for c in df.columns if c in mapping}
        if cols:
            df = df.rename(columns=cols)
            df.to_parquet(p, index=False)


def _episodes_mapping(mapping: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for old, new in mapping.items():
        for s in VIDEO_META_SUFFIXES:
            result[f"videos/{old}/{s}"] = f"videos/{new}/{s}"
    return result


def _rename_episode_parquet_columns(dst: Path, mapping: dict[str, str]) -> None:
    ep_map = _episodes_mapping(mapping)
    for p in sorted((dst / "meta" / "episodes").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(p)
        cols = {c: ep_map[c] for c in df.columns if c in ep_map}
        if cols:
            df = df.rename(columns=cols)
            df.to_parquet(p, index=False)


def _rename_video_dirs(dst: Path, mapping: dict[str, str]) -> None:
    for old, new in mapping.items():
        old_dir = dst / "videos" / old
        new_dir = dst / "videos" / new
        if old_dir.exists():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            if new_dir.exists():
                shutil.rmtree(new_dir)
            old_dir.rename(new_dir)


def _verify(dst: Path, mapping: dict[str, str]) -> None:
    info = json.loads((dst / "meta" / "info.json").read_text(encoding="utf-8"))
    feature_keys = set(info.get("features", {}).keys())

    old_keys = set(mapping.keys())
    new_keys = set(mapping.values())

    still_old = old_keys & feature_keys
    missing_new = new_keys - feature_keys

    if still_old:
        raise RuntimeError(f"Old keys still in info.json: {sorted(still_old)}")
    if missing_new:
        raise RuntimeError(f"New keys missing in info.json: {sorted(missing_new)}")

    for new in sorted(new_keys):
        vdir = dst / "videos" / new
        if not vdir.exists():
            raise RuntimeError(f"Missing video dir: {vdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename LeRobot camera keys consistently.")
    parser.add_argument("--src", type=Path, required=True, help="Source dataset directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination dataset directory")
    parser.add_argument(
        "--mapping-json",
        type=str,
        default=None,
        help="JSON string mapping old->new feature keys",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification on --dst without modifying files",
    )
    args = parser.parse_args()

    mapping = DEFAULT_MAPPING if args.mapping_json is None else json.loads(args.mapping_json)

    if args.verify_only:
        _verify(args.dst, mapping)
        print("VERIFY_OK")
        return

    if not args.src.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.src}")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)

    _rename_json_keys(args.dst / "meta" / "info.json", mapping, top_key="features")

    stats_path = args.dst / "meta" / "stats.json"
    if stats_path.exists():
        _rename_json_keys(stats_path, mapping, top_key=None)

    _rename_data_parquet_columns(args.dst, mapping)
    _rename_episode_parquet_columns(args.dst, mapping)
    _rename_video_dirs(args.dst, mapping)
    _verify(args.dst, mapping)

    print("DONE")
    print(args.dst)


if __name__ == "__main__":
    main()
