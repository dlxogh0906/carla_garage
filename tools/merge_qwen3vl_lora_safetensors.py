#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a Qwen3-VL safetensors checkpoint copy.

This script intentionally does not modify the base model directory.  It reads
the base model shards and the LoRA adapter, then writes a new full-model
safetensors directory with the same shard layout as the base checkpoint.

It is dependency-light on purpose: torch + safetensors are enough, so the merge
does not require installing peft in the quantization environment.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Full Qwen3-VL base model directory.")
    parser.add_argument("--lora-dir", required=True, help="PEFT LoRA checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="New merged full-model output directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and recreate output dir if it already exists. Refuses source dirs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and key mapping without writing the merged model.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve()


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
    # PEFT stores keys as base_model.model.<actual HF weight prefix>.
    for prefix in ("base_model.model.", "base_model."):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return f"{stem}.weight", side


def build_lora_pairs(
    adapter_tensors: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rank = int(adapter_config.get("r", 0) or 0)
    alpha = float(adapter_config.get("lora_alpha", rank) or rank)
    scale = alpha / rank if rank else 1.0
    fan_in_out = bool(adapter_config.get("fan_in_fan_out", False))
    if fan_in_out:
        raise NotImplementedError("fan_in_fan_out=True is not supported by this merge script.")

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


def validate_pairs(weight_map: dict[str, str], pairs: dict[str, dict[str, Any]]) -> None:
    missing = [k for k in pairs if k not in weight_map]
    if missing:
        raise RuntimeError(
            "LoRA target weights not found in base model index. "
            f"First missing keys: {missing[:10]}"
        )


def merge_weight(weight: torch.Tensor, item: dict[str, Any], key: str) -> torch.Tensor:
    lora_a = item["A"].to(dtype=torch.float32)
    lora_b = item["B"].to(dtype=torch.float32)
    delta = torch.matmul(lora_b, lora_a) * float(item["scale"])
    if tuple(delta.shape) != tuple(weight.shape):
        raise RuntimeError(
            f"Shape mismatch for {key}: base={tuple(weight.shape)} delta={tuple(delta.shape)}"
        )
    merged = (weight.to(dtype=torch.float32) + delta).to(dtype=weight.dtype)
    return merged.contiguous()


def main() -> None:
    args = parse_args()
    base_dir = safe_resolve(Path(args.base_model))
    lora_dir = safe_resolve(Path(args.lora_dir))
    output_dir = safe_resolve(Path(args.output_dir))

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base model dir not found: {base_dir}")
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")
    if output_dir in {base_dir, lora_dir}:
        raise RuntimeError("Output dir must be different from base model and LoRA dirs.")

    index_path = base_dir / "model.safetensors.index.json"
    adapter_path = lora_dir / "adapter_model.safetensors"
    adapter_config_path = lora_dir / "adapter_config.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    if not adapter_config_path.exists():
        raise FileNotFoundError(adapter_config_path)

    index = read_json(index_path)
    weight_map: dict[str, str] = dict(index.get("weight_map") or {})
    adapter_config = read_json(adapter_config_path)
    adapter_tensors = load_adapter(adapter_path)
    pairs = build_lora_pairs(adapter_tensors, adapter_config)
    validate_pairs(weight_map, pairs)

    shard_names = sorted(set(weight_map.values()))
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Qwen3-VL LoRA merge")
    print(f"base      : {base_dir}")
    print(f"lora      : {lora_dir}")
    print(f"output    : {output_dir}")
    print(f"shards    : {len(shard_names)}")
    print(f"lora mods : {len(pairs)}")
    print(f"scale     : {float(adapter_config.get('lora_alpha', 0)) / max(1, int(adapter_config.get('r', 1)))}")
    print(f"dry-run   : {args.dry_run}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.dry_run:
        print("dry-run ok: all LoRA target weights exist in the base model index")
        return

    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise RuntimeError(f"Output dir exists and is not empty: {output_dir}")
        if args.overwrite:
            if output_dir in {Path("/"), base_dir, lora_dir}:
                raise RuntimeError(f"Refusing to remove unsafe output dir: {output_dir}")
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
                    print(f"  + LoRA {key}")
                    tensor = merge_weight(tensor, pairs[key], key)
                    merged_keys.append(key)
                tensors[key] = tensor
        save_file(tensors, str(dst_shard), metadata=metadata)

    if len(merged_keys) != len(pairs):
        missing = sorted(set(pairs) - set(merged_keys))
        raise RuntimeError(f"Not all LoRA weights were merged. Missing: {missing[:10]}")

    metadata = {
        "base_model": str(base_dir),
        "lora_dir": str(lora_dir),
        "output_dir": str(output_dir),
        "adapter_model": str(adapter_path),
        "num_merged_modules": len(merged_keys),
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "lora_scale": float(adapter_config.get("lora_alpha", 0)) / max(1, int(adapter_config.get("r", 1))),
        "target_modules": adapter_config.get("target_modules"),
        "merged_keys": merged_keys,
    }
    with (output_dir / "merge_lora_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"merged full model saved: {output_dir}")
    print("base model was not modified")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
