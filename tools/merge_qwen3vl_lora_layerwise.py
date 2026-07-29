#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into Qwen3-VL with layer/group-specific scales."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


WEIGHT_FILE_RE = re.compile(r"model-\d{5}-of-\d{5}\.safetensors$|model\.safetensors$")
LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--early-end", type=int, default=11)
    parser.add_argument("--middle-end", type=int, default=23)
    parser.add_argument(
        "--group-alphas",
        default="early=0.5,middle=1.0,late=1.25,other=1.0",
        help="Comma-separated group scales, e.g. early=0.5,middle=1.0,late=1.25.",
    )
    parser.add_argument("--alpha-json", help="Optional JSON mapping layer index or base weight key to merge alpha.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_group_alphas(text: str) -> dict[str, float]:
    values = {"early": 1.0, "middle": 1.0, "late": 1.0, "other": 1.0}
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        name, raw = chunk.split("=", 1)
        values[name.strip()] = float(raw)
    return values


def copy_base_sidecars(base_dir: Path, output_dir: Path) -> None:
    for src in base_dir.iterdir():
        if src.is_dir():
            continue
        if src.name == "model.safetensors.index.json":
            continue
        if WEIGHT_FILE_RE.match(src.name):
            continue
        shutil.copy2(src, output_dir / src.name)


def load_adapter(adapter_path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def adapter_key_to_base_weight(key: str) -> tuple[str, str] | None:
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    if key.endswith(suffix_a):
        suffix = suffix_a
        side = "A"
    elif key.endswith(suffix_b):
        suffix = suffix_b
        side = "B"
    else:
        return None

    stem = key[: -len(suffix)]
    for prefix in ("base_model.model.", "base_model."):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return f"{stem}.weight", side


def build_lora_pairs(adapter_tensors: dict[str, torch.Tensor], adapter_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rank = int(adapter_config.get("r", 0) or 0)
    alpha = float(adapter_config.get("lora_alpha", rank) or rank)
    scale = alpha / rank if rank else 1.0
    if bool(adapter_config.get("fan_in_fan_out", False)):
        raise NotImplementedError("fan_in_fan_out=True is not supported.")

    pairs: dict[str, dict[str, Any]] = {}
    for key, tensor in adapter_tensors.items():
        mapped = adapter_key_to_base_weight(key)
        if mapped is None:
            continue
        base_key, side = mapped
        item = pairs.setdefault(base_key, {"scale": scale, "adapter_keys": {}})
        item[side] = tensor
        item["adapter_keys"][side] = key

    incomplete = [k for k, v in pairs.items() if "A" not in v or "B" not in v]
    if incomplete:
        raise RuntimeError(f"Incomplete LoRA A/B pairs: {incomplete[:10]}")
    return pairs


def layer_index(key: str) -> int:
    match = LAYER_RE.search(key)
    if not match:
        return -1
    return int(match.group(1))


def group_for_layer(layer: int, early_end: int, middle_end: int) -> str:
    if layer < 0:
        return "other"
    if layer <= early_end:
        return "early"
    if layer <= middle_end:
        return "middle"
    return "late"


def alpha_for_key(
    key: str,
    group_alphas: dict[str, float],
    alpha_overrides: dict[str, Any],
    early_end: int,
    middle_end: int,
) -> tuple[str, int, float]:
    layer = layer_index(key)
    group = group_for_layer(layer, early_end, middle_end)
    if key in alpha_overrides:
        return group, layer, float(alpha_overrides[key])
    if str(layer) in alpha_overrides:
        return group, layer, float(alpha_overrides[str(layer)])
    return group, layer, float(group_alphas.get(group, 1.0))


def merge_weight(weight: torch.Tensor, item: dict[str, Any], key: str, merge_alpha: float) -> torch.Tensor:
    lora_a = item["A"].to(dtype=torch.float32)
    lora_b = item["B"].to(dtype=torch.float32)
    delta = torch.matmul(lora_b, lora_a) * float(item["scale"]) * merge_alpha
    if tuple(delta.shape) != tuple(weight.shape):
        raise RuntimeError(f"Shape mismatch for {key}: base={tuple(weight.shape)} delta={tuple(delta.shape)}")
    return (weight.to(dtype=torch.float32) + delta).to(dtype=weight.dtype).contiguous()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_model).expanduser().resolve()
    lora_dir = Path(args.lora_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir in {base_dir, lora_dir, Path("/")}:
        raise RuntimeError("Unsafe output directory.")

    index_path = base_dir / "model.safetensors.index.json"
    adapter_path = lora_dir / "adapter_model.safetensors"
    adapter_config_path = lora_dir / "adapter_config.json"
    index = read_json(index_path)
    weight_map: dict[str, str] = dict(index.get("weight_map") or {})
    adapter_config = read_json(adapter_config_path)
    pairs = build_lora_pairs(load_adapter(adapter_path), adapter_config)
    missing = [key for key in pairs if key not in weight_map]
    if missing:
        raise RuntimeError(f"LoRA target weights not found in base model index: {missing[:10]}")

    group_alphas = parse_group_alphas(args.group_alphas)
    alpha_overrides = read_json(Path(args.alpha_json).expanduser().resolve()) if args.alpha_json else {}
    alpha_plan = {
        key: alpha_for_key(key, group_alphas, alpha_overrides, args.early_end, args.middle_end)
        for key in sorted(pairs)
    }
    shard_names = sorted(set(weight_map.values()))

    print("Qwen3-VL layer-wise LoRA merge")
    print(f"base         : {base_dir}")
    print(f"lora         : {lora_dir}")
    print(f"output       : {output_dir}")
    print(f"group alphas : {group_alphas}")
    print(f"layers       : early 0-{args.early_end}, middle {args.early_end + 1}-{args.middle_end}, late {args.middle_end + 1}+")
    print(f"lora modules : {len(pairs)}")
    print(f"dry-run      : {args.dry_run}")

    if args.dry_run:
        counts: dict[str, int] = {}
        for group, _layer, _alpha in alpha_plan.values():
            counts[group] = counts.get(group, 0) + 1
        print(f"alpha plan ok: {counts}")
        return

    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise RuntimeError(f"Output dir exists and is not empty: {output_dir}")
        if args.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_base_sidecars(base_dir, output_dir)
    shutil.copy2(index_path, output_dir / index_path.name)
    shutil.copy2(adapter_config_path, output_dir / "merged_lora_adapter_config.json")

    merged_keys: list[str] = []
    for shard_name in shard_names:
        src_shard = base_dir / shard_name
        dst_shard = output_dir / shard_name
        shard_keys = [k for k, v in weight_map.items() if v == shard_name]
        print(f"[merge] {shard_name}: tensors={len(shard_keys)}")
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(src_shard), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key in shard_keys:
                tensor = f.get_tensor(key)
                if key in pairs:
                    group, layer, merge_alpha = alpha_plan[key]
                    print(f"  + LoRA {key} group={group} layer={layer} merge_alpha={merge_alpha}")
                    tensor = merge_weight(tensor, pairs[key], key, merge_alpha)
                    merged_keys.append(key)
                tensors[key] = tensor
        save_file(tensors, str(dst_shard), metadata=metadata)

    if len(merged_keys) != len(pairs):
        missing_merge = sorted(set(pairs) - set(merged_keys))
        raise RuntimeError(f"Not all LoRA weights were merged. Missing: {missing_merge[:10]}")

    metadata = {
        "base_model": str(base_dir),
        "lora_dir": str(lora_dir),
        "output_dir": str(output_dir),
        "num_merged_modules": len(merged_keys),
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "lora_scale": float(adapter_config.get("lora_alpha", 0)) / max(1, int(adapter_config.get("r", 1))),
        "early_end": args.early_end,
        "middle_end": args.middle_end,
        "group_alphas": group_alphas,
        "alpha_overrides": alpha_overrides,
        "alpha_plan": {
            key: {"group": group, "layer": layer, "merge_alpha": alpha}
            for key, (group, layer, alpha) in alpha_plan.items()
        },
        "merged_keys": merged_keys,
    }
    with (output_dir / "merge_lora_layerwise_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"merged full model saved: {output_dir}")


if __name__ == "__main__":
    main()
