#!/usr/bin/env python3
"""Collect Qwen3-VL activation statistics on driving calibration data.

This is a lightweight sensitivity proxy for mixed-bit quantization:
high-activation and high-outlier modules/layers are good candidates for W8A16
or BF16 protection, while lower-ranked language layers can stay W4A16 GPTQ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from quantize_qwen3vl_w4a16 import (
    DEFAULT_CALIB,
    DEFAULT_MODEL,
    dtype_from_name,
    parse_max_memory,
    pick_model_class,
)


TEXT_LINEAR_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
PROJECTION_RE = re.compile(
    r"^model\.visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]$"
)
VISUAL_BLOCK_RE = re.compile(
    r"^model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])$"
)


@dataclass
class ModuleStats:
    name: str
    group: str
    layer: int | None
    in_features: int
    out_features: int
    param_count: int
    token_vectors: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0
    channel_sum_abs: torch.Tensor | None = None
    channel_sum_sq: torch.Tensor | None = None

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0 or x.shape[-1] != self.in_features:
            return
        flat = x.detach().reshape(-1, x.shape[-1])
        if flat.numel() == 0:
            return
        flat32 = flat.float()
        abs_flat = flat32.abs()
        channel_abs = abs_flat.sum(dim=0).cpu()
        channel_sq = flat32.square().sum(dim=0).cpu()
        if self.channel_sum_abs is None:
            self.channel_sum_abs = torch.zeros_like(channel_abs)
            self.channel_sum_sq = torch.zeros_like(channel_sq)
        self.channel_sum_abs += channel_abs
        self.channel_sum_sq += channel_sq
        self.token_vectors += int(flat32.shape[0])
        self.sum_abs += float(channel_abs.sum().item())
        self.sum_sq += float(channel_sq.sum().item())
        self.max_abs = max(self.max_abs, float(abs_flat.max().item()))

    def to_row(self, top_channels: int) -> dict[str, Any]:
        denom = max(1, self.token_vectors * self.in_features)
        mean_abs = self.sum_abs / denom
        rms = math.sqrt(self.sum_sq / denom)
        outlier_ratio = self.max_abs / max(mean_abs, 1e-12)

        top_channel_ids: list[int] = []
        top_channel_mean_abs: list[float] = []
        top_channel_rms: list[float] = []
        if self.channel_sum_abs is not None and top_channels > 0:
            k = min(top_channels, int(self.channel_sum_abs.numel()))
            values, indices = torch.topk(self.channel_sum_abs, k)
            top_channel_ids = [int(i) for i in indices.tolist()]
            top_channel_mean_abs = [
                float(v / max(1, self.token_vectors)) for v in values.tolist()
            ]
            assert self.channel_sum_sq is not None
            top_channel_rms = [
                float(math.sqrt(float(self.channel_sum_sq[i].item()) / max(1, self.token_vectors)))
                for i in top_channel_ids
            ]

        return {
            "name": self.name,
            "group": self.group,
            "layer": "" if self.layer is None else self.layer,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "param_count": self.param_count,
            "token_vectors": self.token_vectors,
            "mean_abs": mean_abs,
            "rms": rms,
            "max_abs": self.max_abs,
            "outlier_ratio": outlier_ratio,
            "top_channel_ids": top_channel_ids,
            "top_channel_mean_abs": top_channel_mean_abs,
            "top_channel_rms": top_channel_rms,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--calib-jsonl", default=DEFAULT_CALIB)
    parser.add_argument(
        "--output-json",
        default="/mnt/2/carla_metric_result/qwen_activation_importance/qwen3vl_activation_importance.json",
    )
    parser.add_argument(
        "--output-csv",
        default="/mnt/2/carla_metric_result/qwen_activation_importance/qwen3vl_activation_importance.csv",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", default="")
    parser.add_argument("--offload-folder", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--top-channels", type=int, default=16)
    parser.add_argument(
        "--target-scope",
        choices=["language", "language_projection", "all"],
        default="language_projection",
    )
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--w8-top-layers", type=int, default=4)
    parser.add_argument(
        "--always-w8-boundary-layers",
        type=int,
        default=1,
        help="Always include this many first and last language layers in the W8 suggestion.",
    )
    return parser.parse_args()


def read_calibration_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = Path(item["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            rows.append(
                {
                    "text": item["chat_template_text"],
                    "image_path": str(image_path),
                    "sample_id": item.get("sample_id", ""),
                }
            )
            if len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No calibration rows found in {path}")
    return rows


def classify_module(name: str, include_lm_head: bool) -> tuple[str, int | None] | None:
    match = TEXT_LINEAR_RE.match(name)
    if match:
        return "language", int(match.group(1))
    if PROJECTION_RE.match(name):
        return "projection", None
    if VISUAL_BLOCK_RE.match(name):
        return "visual_block", None
    if include_lm_head and name == "lm_head":
        return "lm_head", None
    return None


def selected_by_scope(group: str, scope: str) -> bool:
    if scope == "language":
        return group == "language"
    if scope == "language_projection":
        return group in {"language", "projection", "lm_head"}
    return group in {"language", "projection", "visual_block", "lm_head"}


def collect_modules(
    model: torch.nn.Module,
    scope: str,
    include_lm_head: bool,
) -> dict[str, ModuleStats]:
    stats: dict[str, ModuleStats] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        classified = classify_module(name, include_lm_head)
        if classified is None:
            continue
        group, layer = classified
        if not selected_by_scope(group, scope):
            continue
        stats[name] = ModuleStats(
            name=name,
            group=group,
            layer=layer,
            in_features=int(module.in_features),
            out_features=int(module.out_features),
            param_count=int(module.in_features * module.out_features),
        )
    if not stats:
        raise RuntimeError("No target Linear modules matched.")
    return stats


def register_hooks(model: torch.nn.Module, stats: dict[str, ModuleStats]) -> list[Any]:
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            stats[name].update(inputs[0])

        return hook

    for name, module in model.named_modules():
        if name in stats:
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    return handles


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def move_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs


def add_importance_scores(rows: list[dict[str, Any]]) -> None:
    max_rms = max((float(row["rms"]) for row in rows), default=1.0) or 1.0
    max_abs = max((float(row["max_abs"]) for row in rows), default=1.0) or 1.0
    max_outlier = max((float(row["outlier_ratio"]) for row in rows), default=1.0) or 1.0
    max_params = max((float(row["param_count"]) for row in rows), default=1.0) or 1.0
    for row in rows:
        score = (
            0.40 * float(row["rms"]) / max_rms
            + 0.30 * float(row["max_abs"]) / max_abs
            + 0.20 * float(row["outlier_ratio"]) / max_outlier
            + 0.10 * float(row["param_count"]) / max_params
        )
        row["importance_score"] = score


def summarize_layers(rows: list[dict[str, Any]], num_layers: int, top_k: int, boundary: int) -> dict[str, Any]:
    layer_scores: dict[int, float] = {idx: 0.0 for idx in range(num_layers)}
    layer_counts: dict[int, int] = {idx: 0 for idx in range(num_layers)}
    for row in rows:
        if row["group"] != "language" or row["layer"] == "":
            continue
        layer = int(row["layer"])
        layer_scores[layer] += float(row["importance_score"])
        layer_counts[layer] += 1
    layer_summary = []
    for idx in range(num_layers):
        avg_score = layer_scores[idx] / max(1, layer_counts[idx])
        layer_summary.append({"layer": idx, "score": avg_score, "modules": layer_counts[idx]})

    ranked = sorted(layer_summary, key=lambda item: item["score"], reverse=True)
    suggested = {int(item["layer"]) for item in ranked[: max(0, top_k)]}
    for idx in range(max(0, boundary)):
        if idx < num_layers:
            suggested.add(idx)
        last = num_layers - 1 - idx
        if last >= 0:
            suggested.add(last)

    return {
        "layer_summary": layer_summary,
        "ranked_layers": ranked,
        "suggested_w8_layers": sorted(suggested),
        "suggested_w8_layers_arg": ",".join(str(idx) for idx in sorted(suggested)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "name",
        "group",
        "layer",
        "importance_score",
        "in_features",
        "out_features",
        "param_count",
        "token_vectors",
        "mean_abs",
        "rms",
        "max_abs",
        "outlier_ratio",
        "top_channel_ids",
        "top_channel_mean_abs",
        "top_channel_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = {key: row.get(key, "") for key in fieldnames}
            out["rank"] = idx
            for key in ("top_channel_ids", "top_channel_mean_abs", "top_channel_rms"):
                out[key] = json.dumps(out[key])
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    calib_path = Path(args.calib_jsonl)
    rows = read_calibration_rows(calib_path, args.num_samples)

    print("========================================")
    print("Qwen3-VL activation importance analysis")
    print(f"model       : {model_path}")
    print(f"calib       : {calib_path}")
    print(f"samples     : {len(rows)}")
    print(f"scope       : {args.target_scope}")
    print("========================================")

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, use_fast=True)
    cfg, model_cls, class_name = pick_model_class(str(model_path))
    load_kwargs = {
        "dtype": dtype_from_name(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    max_memory = parse_max_memory(args.max_memory)
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if args.offload_folder:
        load_kwargs["offload_folder"] = args.offload_folder

    model = model_cls.from_pretrained(str(model_path), **load_kwargs)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    stats = collect_modules(model, args.target_scope, args.include_lm_head)
    print(f"matched modules: {len(stats)}")
    handles = register_hooks(model, stats)

    device = first_parameter_device(model)
    try:
        for idx, item in enumerate(rows, start=1):
            image = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(text=[item["text"]], images=[image], return_tensors="pt")
            inputs = move_inputs_to_device(inputs, device)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            print(f"[{idx}/{len(rows)}] {item['sample_id'] or item['image_path']}")
    finally:
        for handle in handles:
            handle.remove()

    result_rows = [module_stats.to_row(args.top_channels) for module_stats in stats.values()]
    add_importance_scores(result_rows)
    result_rows.sort(key=lambda row: float(row["importance_score"]), reverse=True)

    text_cfg = getattr(cfg, "text_config", None)
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or getattr(cfg, "num_hidden_layers", 0))
    layer_report = summarize_layers(
        result_rows,
        num_layers=num_layers,
        top_k=args.w8_top_layers,
        boundary=args.always_w8_boundary_layers,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": str(model_path),
        "model_class": class_name,
        "calibration_jsonl": str(calib_path),
        "num_samples": len(rows),
        "target_scope": args.target_scope,
        "importance_score": {
            "description": "Activation proxy: 0.40*rms + 0.30*max_abs + 0.20*outlier_ratio + 0.10*param_count after max normalization.",
        },
        **layer_report,
        "modules": result_rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(output_csv, result_rows)

    print("========================================")
    print(f"wrote JSON: {output_json}")
    print(f"wrote CSV : {output_csv}")
    print(f"suggested W8 layers: {layer_report['suggested_w8_layers_arg']}")
    print("use with:")
    print(
        "  tools/quantize_qwen3vl_mixed_gptq.py "
        f"--w8-layers {layer_report['suggested_w8_layers_arg']}"
    )
    print("top modules:")
    for row in result_rows[:10]:
        print(f"  {row['importance_score']:.4f}  {row['name']}")
    print("========================================")


if __name__ == "__main__":
    main()
