"""Create a static report for fine-tuning weight deltas.

The report treats each task model as a delta from a base checkpoint:

    delta = task_weight - base_weight

It summarizes those deltas by tensor, module group, MLP neuron, and attention
head candidates. The result is a hypothesis generator, not causal proof that a
unit implements a skill. Use activation/ablation tests to validate candidates.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
EPS = 1e-12


@dataclass(frozen=True)
class ModelRef:
    name: str
    repo_id: str
    cache_dir: Path
    snapshot_dir: Path
    safetensors_path: Path
    config_path: Path
    train_config_path: Path
    policy_type: str | None
    policy_repo_id: str | None
    num_vlm_layers: int | None
    pretrained_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a policy weight-delta report from local Hugging Face cache."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face hub cache directory.",
    )
    parser.add_argument(
        "--base",
        default="lerobot/smolvla_base",
        help="Base model repo id, cache folder name, snapshot dir, or model.safetensors path.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Task models to compare. If omitted, all cached SmolVLA models are discovered.",
    )
    parser.add_argument(
        "--include-regex",
        default="",
        help="Only include discovered task model names matching this regex.",
    )
    parser.add_argument(
        "--exclude-regex",
        default="32layers",
        help="Exclude discovered task model names matching this regex. Use '' to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "smolvla_weight_diff_report",
        help="Directory for CSV, PNG, and HTML outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of top tensors/neuron candidates to show in the HTML report.",
    )
    parser.add_argument(
        "--sample-per-tensor",
        type=int,
        default=512,
        help="Deterministic samples per tensor for approximate task-delta similarity.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Attention head dimension used to group q/k/v rows and o columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="Absolute delta threshold for changed_fraction.",
    )
    return parser.parse_args()


def repo_id_to_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def cache_name_to_repo_id(name: str) -> str:
    if name.startswith("models--"):
        return name.removeprefix("models--").replace("--", "/")
    return name


def latest_snapshot_dir(model_cache_dir: Path) -> Path | None:
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_main = model_cache_dir / "refs" / "main"
    if ref_main.exists():
        commit = ref_main.read_text(encoding="utf-8").strip()
        if commit and (snapshots_dir / commit).exists():
            return snapshots_dir / commit

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def train_policy_config(train_config: dict) -> dict:
    policy = train_config.get("policy")
    return policy if isinstance(policy, dict) else {}


def model_ref_from_snapshot(model_cache_dir: Path, snapshot_dir: Path) -> ModelRef:
    config_path = snapshot_dir / "config.json"
    train_config_path = snapshot_dir / "train_config.json"
    config = load_config(config_path)
    train_config = load_config(train_config_path)
    train_policy = train_policy_config(train_config)
    repo_id = cache_name_to_repo_id(model_cache_dir.name)
    policy_repo_id = train_policy.get("repo_id") or config.get("repo_id")
    return ModelRef(
        name=model_cache_dir.name,
        repo_id=repo_id,
        cache_dir=model_cache_dir,
        snapshot_dir=snapshot_dir,
        safetensors_path=snapshot_dir / "model.safetensors",
        config_path=config_path,
        train_config_path=train_config_path,
        policy_type=train_policy.get("type") or config.get("type"),
        policy_repo_id=policy_repo_id,
        num_vlm_layers=train_policy.get("num_vlm_layers") or config.get("num_vlm_layers"),
        pretrained_path=train_policy.get("pretrained_path") or config.get("pretrained_path"),
    )


def model_ref_from_cache_dir(model_cache_dir: Path) -> ModelRef | None:
    snapshot_dir = latest_snapshot_dir(model_cache_dir)
    if snapshot_dir is None:
        return None
    safetensors_path = snapshot_dir / "model.safetensors"
    if not safetensors_path.exists():
        return None
    return model_ref_from_snapshot(model_cache_dir, snapshot_dir)


def resolve_model_ref(value: str, cache_dir: Path) -> ModelRef:
    path = Path(value)
    if path.exists():
        if path.is_file():
            snapshot_dir = path.parent
            model_cache_dir = snapshot_dir.parent.parent
        elif (path / "model.safetensors").exists():
            snapshot_dir = path
            model_cache_dir = snapshot_dir.parent.parent
        else:
            model_cache_dir = path
            ref = model_ref_from_cache_dir(model_cache_dir)
            if ref is None:
                raise FileNotFoundError(f"Could not resolve model cache directory: {value}")
            return ref

        return model_ref_from_snapshot(model_cache_dir, snapshot_dir)

    model_cache_dir = cache_dir / repo_id_to_cache_name(value)
    ref = model_ref_from_cache_dir(model_cache_dir)
    if ref is None:
        model_cache_dir = cache_dir / value
        ref = model_ref_from_cache_dir(model_cache_dir)
    if ref is None:
        raise FileNotFoundError(f"Could not find cached model for {value!r} in {cache_dir}")
    return ref


def discover_smolvla_models(
    cache_dir: Path,
    base_ref: ModelRef,
    include_regex: str,
    exclude_regex: str,
) -> list[ModelRef]:
    include_re = re.compile(include_regex, re.IGNORECASE) if include_regex else None
    exclude_re = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
    refs: list[ModelRef] = []

    for model_cache_dir in sorted(cache_dir.glob("models--*")):
        ref = model_ref_from_cache_dir(model_cache_dir)
        if ref is None:
            continue
        config = load_config(ref.config_path)
        haystack = " ".join(
            [
                ref.name,
                ref.repo_id or "",
                ref.policy_repo_id or "",
                ref.policy_type or "",
                str(config.get("type", "")),
                str(config.get("vlm_model_name", "")),
            ]
        )
        if ref.policy_type != "smolvla" and "smolvla" not in haystack.lower():
            continue
        if ref.safetensors_path == base_ref.safetensors_path:
            continue
        if include_re and not include_re.search(haystack):
            continue
        if exclude_re and exclude_re.search(haystack):
            continue
        refs.append(ref)

    return refs


def group_name(tensor_name: str) -> str:
    if tensor_name.startswith(("model.action_", "action_", "action_in_proj", "action_out_proj")):
        return "action_head"
    if tensor_name.startswith(("model.state_proj", "state_proj")):
        return "state_proj"
    if tensor_name.startswith("time_mlp_"):
        return "time_mlp"

    expert = re.search(r"(?:lm_expert\.layers|gemma_expert\.model\.layers)\.(\d+)\.", tensor_name)
    if expert:
        layer = int(expert.group(1))
        if ".self_attn." in tensor_name:
            return f"expert_attn_L{layer:02d}"
        if ".mlp." in tensor_name:
            return f"expert_mlp_L{layer:02d}"
        return f"expert_other_L{layer:02d}"

    vlm = re.search(
        r"(?:model\.vlm_with_expert\.vlm\.model\.layers|paligemma\.model\.language_model\.layers)\.(\d+)\.",
        tensor_name,
    )
    if vlm:
        layer = int(vlm.group(1))
        if ".self_attn." in tensor_name:
            return f"vlm_attn_L{layer:02d}"
        if ".mlp." in tensor_name:
            return f"vlm_mlp_L{layer:02d}"
        return f"vlm_other_L{layer:02d}"

    vision_layer = re.search(r"vision_model\.encoder\.layers\.(\d+)\.", tensor_name)
    if vision_layer:
        layer = int(vision_layer.group(1))
        if ".self_attn." in tensor_name:
            return f"vision_attn_L{layer:02d}"
        if ".mlp." in tensor_name:
            return f"vision_mlp_L{layer:02d}"
        return f"vision_other_L{layer:02d}"

    if "multi_modal_projector" in tensor_name:
        return "multi_modal_projector"
    if "vision_model" in tensor_name or "vision_tower" in tensor_name:
        return "vision_encoder"
    if "embed" in tensor_name:
        return "embeddings"
    if "lm_head" in tensor_name:
        return "lm_head"
    return "other"


def canonical_tensor_key(key: str) -> str:
    return key.removeprefix("model.")


def tensor_key_map(keys: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in keys:
        canonical = canonical_tensor_key(key)
        result.setdefault(canonical, key)
    return result


def tensor_stats(delta: torch.Tensor, base: torch.Tensor, threshold: float) -> dict[str, float | int]:
    delta_f = delta.float()
    base_f = base.float()
    delta_sq = float(torch.sum(delta_f * delta_f).item())
    base_sq = float(torch.sum(base_f * base_f).item())
    abs_delta = torch.abs(delta_f)
    numel = delta_f.numel()
    return {
        "numel": numel,
        "delta_l2": math.sqrt(delta_sq),
        "base_l2": math.sqrt(base_sq),
        "relative_l2": math.sqrt(delta_sq) / (math.sqrt(base_sq) + EPS),
        "mean_abs_delta": float(torch.mean(abs_delta).item()),
        "max_abs_delta": float(torch.max(abs_delta).item()),
        "changed_fraction": float(torch.mean((abs_delta > threshold).float()).item()),
        "delta_sq": delta_sq,
        "base_sq": base_sq,
    }


def deterministic_sample(tensor: torch.Tensor, max_items: int) -> np.ndarray:
    if max_items <= 0:
        return np.empty((0,), dtype=np.float32)
    flat = tensor.detach().float().flatten()
    numel = flat.numel()
    if numel == 0:
        return np.empty((0,), dtype=np.float32)
    if numel <= max_items:
        sampled = flat
    else:
        idx = torch.linspace(0, numel - 1, steps=max_items, dtype=torch.long)
        sampled = flat[idx]
    return sampled.cpu().numpy().astype(np.float32, copy=False)


def aggregate_groups(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["model", "repo_id", "group"], as_index=False)
        .agg(
            numel=("numel", "sum"),
            delta_sq=("delta_sq", "sum"),
            base_sq=("base_sq", "sum"),
            mean_abs_delta=("mean_abs_delta", "mean"),
            max_abs_delta=("max_abs_delta", "max"),
            changed_fraction=("changed_fraction", "mean"),
        )
        .copy()
    )
    agg["delta_l2"] = np.sqrt(agg["delta_sq"])
    agg["base_l2"] = np.sqrt(agg["base_sq"])
    agg["relative_l2"] = agg["delta_l2"] / (agg["base_l2"] + EPS)
    return agg.drop(columns=["delta_sq", "base_sq"])


def compare_one_model(
    base_ref: ModelRef,
    task_ref: ModelRef,
    threshold: float,
    sample_per_tensor: int,
) -> tuple[list[dict], dict[str, np.ndarray], list[str]]:
    rows: list[dict] = []
    signature_parts: dict[str, list[np.ndarray]] = {"global": []}
    skipped: list[str] = []

    with safe_open(base_ref.safetensors_path, framework="pt", device="cpu") as base_file:
        with safe_open(task_ref.safetensors_path, framework="pt", device="cpu") as task_file:
            base_key_map = tensor_key_map(base_file.keys())
            task_key_map = tensor_key_map(task_file.keys())
            common_keys = sorted(set(base_key_map) & set(task_key_map))

            for key in common_keys:
                base_key = base_key_map[key]
                task_key = task_key_map[key]
                base_tensor = base_file.get_tensor(base_key)
                task_tensor = task_file.get_tensor(task_key)
                if tuple(base_tensor.shape) != tuple(task_tensor.shape):
                    skipped.append(key)
                    continue

                delta = task_tensor.float() - base_tensor.float()
                stats = tensor_stats(delta, base_tensor, threshold)
                group = group_name(key)
                rows.append(
                    {
                        "model": task_ref.name,
                        "repo_id": task_ref.repo_id,
                        "policy_repo_id": task_ref.policy_repo_id,
                        "policy_type": task_ref.policy_type,
                        "pretrained_path": task_ref.pretrained_path,
                        "num_vlm_layers": task_ref.num_vlm_layers,
                        "tensor": key,
                        "base_tensor": base_key,
                        "task_tensor": task_key,
                        "shape": "x".join(str(v) for v in base_tensor.shape),
                        "group": group,
                        **stats,
                    }
                )

                sample = deterministic_sample(delta, sample_per_tensor)
                signature_parts["global"].append(sample)
                signature_parts.setdefault(group, []).append(sample)

    signatures = {
        name: np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
        for name, parts in signature_parts.items()
    }
    return rows, signatures, skipped


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    size = min(a.size, b.size)
    a2 = a[:size]
    b2 = b[:size]
    denom = float(np.linalg.norm(a2) * np.linalg.norm(b2))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(a2, b2) / denom)


def build_similarity_rows(signatures: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows: list[dict] = []
    model_names = sorted(signatures)
    groups = sorted({g for model_sigs in signatures.values() for g in model_sigs})
    for left, right in combinations(model_names, 2):
        for group in groups:
            if group not in signatures[left] or group not in signatures[right]:
                continue
            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "group": group,
                    "sampled_cosine": cosine(signatures[left][group], signatures[right][group]),
                    "samples": min(signatures[left][group].size, signatures[right][group].size),
                }
            )
    return rows


def get_tensor(file_handle, key: str) -> torch.Tensor | None:
    if key not in set(file_handle.keys()):
        return None
    return file_handle.get_tensor(key).float()


def mlp_neuron_candidates(
    base_ref: ModelRef,
    task_ref: ModelRef,
    top_k: int,
) -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"(.+layers\.(\d+)\.mlp)\.gate_proj\.weight$")

    with safe_open(base_ref.safetensors_path, framework="pt", device="cpu") as base_file:
        with safe_open(task_ref.safetensors_path, framework="pt", device="cpu") as task_file:
            base_key_map = tensor_key_map(base_file.keys())
            task_key_map = tensor_key_map(task_file.keys())
            for gate_key in sorted(k for k in task_key_map if k.endswith(".mlp.gate_proj.weight")):
                match = pattern.search(gate_key)
                if not match:
                    continue
                prefix = match.group(1)
                layer = int(match.group(2))
                up_key = f"{prefix}.up_proj.weight"
                down_key = f"{prefix}.down_proj.weight"
                keys = [gate_key, up_key, down_key]
                if any(k not in task_key_map or k not in base_key_map for k in keys):
                    continue

                gate_base = base_file.get_tensor(base_key_map[gate_key]).float()
                gate_task = task_file.get_tensor(task_key_map[gate_key]).float()
                up_base = base_file.get_tensor(base_key_map[up_key]).float()
                up_task = task_file.get_tensor(task_key_map[up_key]).float()
                down_base = base_file.get_tensor(base_key_map[down_key]).float()
                down_task = task_file.get_tensor(task_key_map[down_key]).float()

                if gate_base.shape != gate_task.shape or up_base.shape != up_task.shape:
                    continue
                if down_base.shape != down_task.shape:
                    continue
                if gate_base.shape[0] != up_base.shape[0] or gate_base.shape[0] != down_base.shape[1]:
                    continue

                gate_delta_sq = torch.sum((gate_task - gate_base) ** 2, dim=1)
                up_delta_sq = torch.sum((up_task - up_base) ** 2, dim=1)
                down_delta_sq = torch.sum((down_task - down_base) ** 2, dim=0)
                gate_base_sq = torch.sum(gate_base**2, dim=1)
                up_base_sq = torch.sum(up_base**2, dim=1)
                down_base_sq = torch.sum(down_base**2, dim=0)

                delta_l2 = torch.sqrt(gate_delta_sq + up_delta_sq + down_delta_sq)
                base_l2 = torch.sqrt(gate_base_sq + up_base_sq + down_base_sq)
                rel = delta_l2 / (base_l2 + EPS)
                limit = min(top_k, rel.numel())
                values, indices = torch.topk(rel, k=limit)

                block = "expert" if "lm_expert" in prefix or "gemma_expert" in prefix else "vlm"
                for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                    rows.append(
                        {
                            "model": task_ref.name,
                            "repo_id": task_ref.repo_id,
                            "rank": rank,
                            "block": block,
                            "layer": layer,
                            "neuron_index": idx,
                            "relative_l2": value,
                            "delta_l2": float(delta_l2[idx].item()),
                            "base_l2": float(base_l2[idx].item()),
                            "gate_key": gate_key,
                            "up_key": up_key,
                            "down_key": down_key,
                        }
                    )
    return rows


def attention_head_candidates(
    base_ref: ModelRef,
    task_ref: ModelRef,
    head_dim: int,
    top_k: int,
) -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"(.+layers\.(\d+)\.self_attn)\.q_proj\.weight$")

    with safe_open(base_ref.safetensors_path, framework="pt", device="cpu") as base_file:
        with safe_open(task_ref.safetensors_path, framework="pt", device="cpu") as task_file:
            base_key_map = tensor_key_map(base_file.keys())
            task_key_map = tensor_key_map(task_file.keys())
            for q_key in sorted(k for k in task_key_map if k.endswith(".self_attn.q_proj.weight")):
                match = pattern.search(q_key)
                if not match:
                    continue
                prefix = match.group(1)
                layer = int(match.group(2))
                keys = [
                    f"{prefix}.q_proj.weight",
                    f"{prefix}.k_proj.weight",
                    f"{prefix}.v_proj.weight",
                    f"{prefix}.o_proj.weight",
                ]
                if any(k not in task_key_map or k not in base_key_map for k in keys):
                    continue

                per_kind: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
                    "query_output": ([], []),
                    "key_value": ([], []),
                }
                head_counts: dict[str, int] = {}
                for key in keys:
                    base = base_file.get_tensor(base_key_map[key]).float()
                    task = task_file.get_tensor(task_key_map[key]).float()
                    if base.shape != task.shape or base.ndim != 2:
                        continue
                    delta = task - base
                    if key.endswith("q_proj.weight"):
                        if base.shape[0] % head_dim != 0:
                            continue
                        n_heads = base.shape[0] // head_dim
                        delta_sq = torch.sum(delta.reshape(n_heads, head_dim, base.shape[1]) ** 2, dim=(1, 2))
                        base_sq = torch.sum(base.reshape(n_heads, head_dim, base.shape[1]) ** 2, dim=(1, 2))
                        kind = "query_output"
                    elif key.endswith("o_proj.weight"):
                        if base.shape[1] % head_dim != 0:
                            continue
                        n_heads = base.shape[1] // head_dim
                        delta_sq = torch.sum(delta.reshape(base.shape[0], n_heads, head_dim) ** 2, dim=(0, 2))
                        base_sq = torch.sum(base.reshape(base.shape[0], n_heads, head_dim) ** 2, dim=(0, 2))
                    else:
                        if base.shape[0] % head_dim != 0:
                            continue
                        n_heads = base.shape[0] // head_dim
                        delta_sq = torch.sum(delta.reshape(n_heads, head_dim, base.shape[1]) ** 2, dim=(1, 2))
                        base_sq = torch.sum(base.reshape(n_heads, head_dim, base.shape[1]) ** 2, dim=(1, 2))
                        kind = "key_value"
                    if key.endswith("o_proj.weight"):
                        kind = "query_output"
                    if kind in head_counts and head_counts[kind] != n_heads:
                        continue
                    head_counts[kind] = n_heads
                    per_kind[kind][0].append(delta_sq)
                    per_kind[kind][1].append(base_sq)

                block = "expert" if "lm_expert" in prefix or "gemma_expert" in prefix else "vlm"
                for kind, (per_head_delta, per_head_base) in per_kind.items():
                    if not per_head_delta:
                        continue
                    if len({x.numel() for x in per_head_delta}) != 1:
                        continue
                    delta_l2 = torch.sqrt(torch.stack(per_head_delta).sum(dim=0))
                    base_l2 = torch.sqrt(torch.stack(per_head_base).sum(dim=0))
                    rel = delta_l2 / (base_l2 + EPS)
                    limit = min(top_k, rel.numel())
                    values, indices = torch.topk(rel, k=limit)

                    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                        rows.append(
                            {
                                "model": task_ref.name,
                                "repo_id": task_ref.repo_id,
                                "rank": rank,
                                "block": block,
                                "layer": layer,
                                "head_kind": kind,
                                "head_index": idx,
                                "head_dim": head_dim,
                                "relative_l2": value,
                                "delta_l2": float(delta_l2[idx].item()),
                                "base_l2": float(base_l2[idx].item()),
                                "prefix": prefix,
                            }
                        )
    return rows


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_lineage_rows(base_ref: ModelRef, task_refs: list[ModelRef]) -> list[dict]:
    rows: list[dict] = []
    ordered_refs = [base_ref, *task_refs]
    for index, ref in enumerate(ordered_refs):
        rows.append(
            {
                "role": "base" if index == 0 else "model",
                "order": index,
                "name": ref.name,
                "repo_id": ref.repo_id,
                "policy_repo_id": ref.policy_repo_id,
                "policy_type": ref.policy_type,
                "pretrained_path": ref.pretrained_path,
                "num_vlm_layers": ref.num_vlm_layers,
                "snapshot_dir": str(ref.snapshot_dir),
                "safetensors_path": str(ref.safetensors_path),
                "config_path": str(ref.config_path),
                "train_config_path": str(ref.train_config_path) if ref.train_config_path.exists() else "",
            }
        )
    return rows


def save_group_heatmap(group_df: pd.DataFrame, output_path: Path) -> None:
    if group_df.empty:
        return
    pivot = group_df.pivot_table(index="model", columns="group", values="relative_l2", aggfunc="max").fillna(0.0)
    ordered_columns = sorted(
        pivot.columns,
        key=lambda c: (0 if c.startswith("action") else 1, c),
    )
    pivot = pivot[ordered_columns]
    values = np.log10(pivot.to_numpy() + 1e-12)

    width = max(12, 0.28 * len(pivot.columns))
    height = max(5, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=80, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Task delta heatmap: log10(relative L2)")
    fig.colorbar(im, ax=ax, label="log10(||delta|| / ||base||)")
    fig.subplots_adjust(left=0.22, right=0.96, top=0.9, bottom=0.42)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_similarity_heatmap(sim_df: pd.DataFrame, output_path: Path) -> None:
    if sim_df.empty:
        return
    global_df = sim_df[sim_df["group"] == "global"]
    if global_df.empty:
        return
    names = sorted(set(global_df["model_a"]) | set(global_df["model_b"]))
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for row in global_df.itertuples(index=False):
        matrix.loc[row.model_a, row.model_b] = row.sampled_cosine
        matrix.loc[row.model_b, row.model_a] = row.sampled_cosine

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), max(5, len(names) * 0.5)))
    im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Approximate task-delta cosine similarity")
    fig.colorbar(im, ax=ax, label="sampled cosine")
    fig.subplots_adjust(left=0.28, right=0.96, top=0.9, bottom=0.34)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def html_table(df: pd.DataFrame, max_rows: int) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    return df.head(max_rows).to_html(index=False, escape=True, classes="data-table")


def write_html_report(
    output_dir: Path,
    base_ref: ModelRef,
    task_refs: list[ModelRef],
    tensor_df: pd.DataFrame,
    group_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    attn_df: pd.DataFrame,
    top_k: int,
) -> None:
    policy_label = base_ref.policy_type or "policy"
    report_title = f"{policy_label.upper()} Weight Delta Report"
    top_tensors = (
        tensor_df.sort_values(["relative_l2"], ascending=False)
        [
            [
                "model",
                "tensor",
                "group",
                "shape",
                "relative_l2",
                "delta_l2",
                "changed_fraction",
            ]
        ]
        .copy()
    )
    top_groups = (
        group_df.sort_values(["relative_l2"], ascending=False)
        [["model", "group", "relative_l2", "delta_l2", "changed_fraction"]]
        .copy()
        if not group_df.empty
        else pd.DataFrame()
    )
    top_similarity = (
        sim_df[sim_df["group"] == "global"].sort_values(["sampled_cosine"], ascending=False).copy()
        if not sim_df.empty
        else pd.DataFrame()
    )
    top_mlp = (
        mlp_df.sort_values(["relative_l2"], ascending=False)
        [["model", "block", "layer", "neuron_index", "relative_l2", "delta_l2"]]
        .copy()
        if not mlp_df.empty
        else pd.DataFrame()
    )
    top_attn = (
        attn_df.sort_values(["relative_l2"], ascending=False)
        [["model", "block", "layer", "head_kind", "head_index", "relative_l2", "delta_l2"]]
        .copy()
        if not attn_df.empty
        else pd.DataFrame()
    )

    model_items = "\n".join(
        f"<li><code>{html.escape(ref.name)}</code> "
        f"repo=<code>{html.escape(ref.repo_id)}</code> "
        f"policy_repo=<code>{html.escape(str(ref.policy_repo_id))}</code> "
        f"(type={html.escape(str(ref.policy_type))}, layers={html.escape(str(ref.num_vlm_layers))}, "
        f"pretrained={html.escape(str(ref.pretrained_path))})</li>"
        for ref in task_refs
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(report_title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; line-height: 1.45; }}
    h1, h2 {{ margin: 1.2em 0 0.4em; }}
    code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 4px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    .note {{ background: #fff8dd; border-left: 4px solid #c9a227; padding: 10px 12px; }}
    .data-table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
    .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 4px 6px; vertical-align: top; }}
    .data-table th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>{html.escape(report_title)}</h1>
  <p>Base: <code>{html.escape(base_ref.repo_id)}</code> (type=<code>{html.escape(str(base_ref.policy_type))}</code>)</p>
  <p class="note">Interpretation: this report ranks <code>W_task - W_base</code>. It is a useful map of fine-tuning pressure and candidate units, but it is not causal evidence that a neuron implements a behavior. Validate candidates with activation probes, dataset contrast, or ablation.</p>

  <h2>Compared Models</h2>
  <ul>
    {model_items}
  </ul>

  <h2>Layer And Module Heatmap</h2>
  <p>Color is <code>log10(||delta|| / ||base||)</code>. Brighter blocks changed more relative to their base norm.</p>
  <img src="group_delta_heatmap.png" alt="group delta heatmap">

  <h2>Task Delta Similarity</h2>
  <p>Approximate cosine similarity from deterministic sampled deltas. High similarity means two task trainings pushed many weights in similar directions.</p>
  <img src="task_similarity_heatmap.png" alt="task similarity heatmap">
  {html_table(top_similarity, top_k)}

  <h2>Top Changed Groups</h2>
  {html_table(top_groups, top_k)}

  <h2>Top Changed Tensors</h2>
  {html_table(top_tensors, top_k)}

  <h2>MLP Neuron Candidates</h2>
  <p>For each MLP, gate/up rows and down columns are combined as one intermediate neuron candidate.</p>
  {html_table(top_mlp, top_k)}

  <h2>Attention Head Candidates</h2>
  <p>Grouped-query attention is split into <code>query_output</code> and <code>key_value</code> head groups, using <code>head_dim</code>.</p>
  {html_table(top_attn, top_k)}

  <h2>Files</h2>
  <ul>
    <li><code>tensor_metrics.csv</code></li>
    <li><code>group_metrics.csv</code></li>
    <li><code>task_similarity.csv</code></li>
    <li><code>mlp_neuron_candidates.csv</code></li>
    <li><code>attention_head_candidates.csv</code></li>
    <li><code>model_lineage.csv</code></li>
  </ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ref = resolve_model_ref(args.base, args.cache_dir)
    if args.models:
        task_refs = [resolve_model_ref(value, args.cache_dir) for value in args.models]
    else:
        task_refs = discover_smolvla_models(
            cache_dir=args.cache_dir,
            base_ref=base_ref,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
        )

    base_layers = base_ref.num_vlm_layers
    comparable_refs = [ref for ref in task_refs if ref.num_vlm_layers == base_layers]
    skipped_refs = [ref for ref in task_refs if ref.num_vlm_layers != base_layers]
    if skipped_refs:
        print("Skipped models with different num_vlm_layers than base:")
        for ref in skipped_refs:
            print(f"  {ref.repo_id} layers={ref.num_vlm_layers}")
    task_refs = comparable_refs

    if not task_refs:
        raise SystemExit("No comparable task models found.")

    print(f"Base: {base_ref.repo_id} ({base_ref.safetensors_path})")
    print(f"Comparing {len(task_refs)} task models")

    all_tensor_rows: list[dict] = []
    all_signatures: dict[str, dict[str, np.ndarray]] = {}
    all_mlp_rows: list[dict] = []
    all_attn_rows: list[dict] = []
    skipped_shapes: list[dict] = []

    for ref in task_refs:
        print(f"Comparing {ref.repo_id}")
        rows, signatures, skipped = compare_one_model(
            base_ref=base_ref,
            task_ref=ref,
            threshold=args.threshold,
            sample_per_tensor=args.sample_per_tensor,
        )
        all_tensor_rows.extend(rows)
        all_signatures[ref.name] = signatures
        skipped_shapes.extend({"model": ref.name, "tensor": key} for key in skipped)
        all_mlp_rows.extend(mlp_neuron_candidates(base_ref, ref, args.top_k))
        all_attn_rows.extend(attention_head_candidates(base_ref, ref, args.head_dim, args.top_k))

    tensor_df = pd.DataFrame(all_tensor_rows)
    group_df = aggregate_groups(all_tensor_rows)
    sim_df = pd.DataFrame(build_similarity_rows(all_signatures))
    mlp_df = pd.DataFrame(all_mlp_rows)
    attn_df = pd.DataFrame(all_attn_rows)

    tensor_df.to_csv(output_dir / "tensor_metrics.csv", index=False)
    group_df.to_csv(output_dir / "group_metrics.csv", index=False)
    sim_df.to_csv(output_dir / "task_similarity.csv", index=False)
    mlp_df.to_csv(output_dir / "mlp_neuron_candidates.csv", index=False)
    attn_df.to_csv(output_dir / "attention_head_candidates.csv", index=False)
    write_csv(output_dir / "model_lineage.csv", model_lineage_rows(base_ref, task_refs))
    write_csv(output_dir / "skipped_shape_mismatches.csv", skipped_shapes)

    save_group_heatmap(group_df, output_dir / "group_delta_heatmap.png")
    save_similarity_heatmap(sim_df, output_dir / "task_similarity_heatmap.png")
    write_html_report(
        output_dir=output_dir,
        base_ref=base_ref,
        task_refs=task_refs,
        tensor_df=tensor_df,
        group_df=group_df,
        sim_df=sim_df,
        mlp_df=mlp_df,
        attn_df=attn_df,
        top_k=args.top_k,
    )

    print(f"Report written to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
