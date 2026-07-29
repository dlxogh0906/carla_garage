#!/usr/bin/env python3
"""Analyze layer/module-wise LoRA update norms for Qwen3-VL PEFT adapters."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.")
MODULE_RE = re.compile(r"\.([^.\s]+)\.weight$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Full base model directory with safetensors index.")
    parser.add_argument("--lora-dir", required=True, help="PEFT LoRA adapter directory.")
    parser.add_argument("--output-prefix", required=True, help="Output path prefix, without extension.")
    parser.add_argument("--early-end", type=int, default=11, help="Last layer index in early group.")
    parser.add_argument("--middle-end", type=int, default=23, help="Last layer index in middle group.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_adapter(adapter_path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def adapter_key_to_base_weight(key: str) -> tuple[str, str] | None:
    suffixes = {".lora_A.weight": "A", ".lora_B.weight": "B"}
    side = None
    suffix = ""
    for candidate, candidate_side in suffixes.items():
        if key.endswith(candidate):
            side = candidate_side
            suffix = candidate
            break
    if side is None:
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


def module_name(key: str) -> str:
    match = MODULE_RE.search(key)
    if not match:
        return "unknown"
    return match.group(1)


def group_name(layer: int, early_end: int, middle_end: int) -> str:
    if layer < 0:
        return "other"
    if layer <= early_end:
        return "early"
    if layer <= middle_end:
        return "middle"
    return "late"


def load_base_weight_norms(base_dir: Path, weight_map: dict[str, str], keys: list[str]) -> dict[str, float]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)

    norms: dict[str, float] = {}
    for shard_name, shard_keys in sorted(by_shard.items()):
        with safe_open(str(base_dir / shard_name), framework="pt", device="cpu") as f:
            for key in shard_keys:
                tensor = f.get_tensor(key).to(dtype=torch.float32)
                norms[key] = float(torch.linalg.vector_norm(tensor).item())
    return norms


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_model).expanduser().resolve()
    lora_dir = Path(args.lora_dir).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    index_path = base_dir / "model.safetensors.index.json"
    adapter_path = lora_dir / "adapter_model.safetensors"
    adapter_config_path = lora_dir / "adapter_config.json"
    index = read_json(index_path)
    weight_map: dict[str, str] = dict(index.get("weight_map") or {})
    adapter_config = read_json(adapter_config_path)
    pairs = build_lora_pairs(load_adapter(adapter_path), adapter_config)

    missing = [key for key in pairs if key not in weight_map]
    if missing:
        raise RuntimeError(f"LoRA targets missing from base index: {missing[:10]}")

    base_norms = load_base_weight_norms(base_dir, weight_map, sorted(pairs))
    rows: list[dict[str, Any]] = []
    for key, item in sorted(pairs.items()):
        lora_a = item["A"].to(dtype=torch.float32)
        lora_b = item["B"].to(dtype=torch.float32)
        delta = torch.matmul(lora_b, lora_a) * float(item["scale"])
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        base_norm = base_norms[key]
        layer = layer_index(key)
        rows.append(
            {
                "layer": layer,
                "group": group_name(layer, args.early_end, args.middle_end),
                "module": module_name(key),
                "base_key": key,
                "delta_norm": delta_norm,
                "base_norm": base_norm,
                "relative_norm": delta_norm / base_norm if base_norm else 0.0,
                "delta_shape": "x".join(str(x) for x in delta.shape),
            }
        )

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    layer_summary: dict[int, dict[str, Any]] = {}
    group_summary: dict[str, dict[str, Any]] = {}
    module_summary: dict[str, dict[str, Any]] = {}
    for bucket_name, bucket in (("layer", layer_summary), ("group", group_summary), ("module", module_summary)):
        values: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            values[row[bucket_name]].append(row)
        for key, items in values.items():
            total_delta_sq = sum(float(x["delta_norm"]) ** 2 for x in items)
            total_base_sq = sum(float(x["base_norm"]) ** 2 for x in items)
            bucket[key] = {
                "num_modules": len(items),
                "delta_norm_l2_aggregate": total_delta_sq ** 0.5,
                "base_norm_l2_aggregate": total_base_sq ** 0.5,
                "relative_norm_l2_aggregate": (total_delta_sq ** 0.5) / (total_base_sq ** 0.5) if total_base_sq else 0.0,
                "mean_relative_norm": sum(float(x["relative_norm"]) for x in items) / len(items),
                "max_relative_norm": max(float(x["relative_norm"]) for x in items),
            }

    payload = {
        "base_model": str(base_dir),
        "lora_dir": str(lora_dir),
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "lora_scale": float(adapter_config.get("lora_alpha", 0)) / max(1, int(adapter_config.get("r", 1))),
        "early_end": args.early_end,
        "middle_end": args.middle_end,
        "num_lora_modules": len(rows),
        "layer_summary": {str(k): v for k, v in sorted(layer_summary.items())},
        "group_summary": group_summary,
        "module_summary": module_summary,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    top_layers = sorted(layer_summary.items(), key=lambda x: x[1]["relative_norm_l2_aggregate"], reverse=True)[:10]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Qwen3-VL LoRA Layer Norm Analysis\n\n")
        f.write(f"- base: `{base_dir}`\n")
        f.write(f"- lora: `{lora_dir}`\n")
        f.write(f"- LoRA modules: `{len(rows)}`\n")
        f.write(f"- groups: early `0-{args.early_end}`, middle `{args.early_end + 1}-{args.middle_end}`, late `{args.middle_end + 1}+`\n\n")
        f.write("## Group Summary\n\n")
        f.write("| Group | Modules | ΔW L2 | W L2 | Relative L2 | Mean Relative |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for name in ("early", "middle", "late", "other"):
            if name not in group_summary:
                continue
            item = group_summary[name]
            f.write(
                f"| {name} | {item['num_modules']} | {item['delta_norm_l2_aggregate']:.6f} | "
                f"{item['base_norm_l2_aggregate']:.6f} | {item['relative_norm_l2_aggregate']:.8f} | "
                f"{item['mean_relative_norm']:.8f} |\n"
            )
        f.write("\n## Top Layers By Relative L2\n\n")
        f.write("| Layer | Modules | Relative L2 | Mean Relative | Max Relative |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: |\n")
        for layer, item in top_layers:
            f.write(
                f"| {layer} | {item['num_modules']} | {item['relative_norm_l2_aggregate']:.8f} | "
                f"{item['mean_relative_norm']:.8f} | {item['max_relative_norm']:.8f} |\n"
            )

    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
