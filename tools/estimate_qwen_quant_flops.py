#!/usr/bin/env python3
"""Estimate dense-equivalent decode FLOPs for Qwen quantization variants.

The reported FLOPs are dense-equivalent decode FLOPs:
  FLOPs/token ~= 2 * dense_parameter_count

Quantized checkpoints can reduce storage, memory traffic, and latency without
changing this dense-equivalent operation count.  The script also reports a
bit-operation proxy so W8A8/W4A16 differences are visible in a separate column.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODELS = [
    ("BF16", "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct", 16),
    ("W8A8-INT8", "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-W8A8-INT8-vllm012", 8),
    ("AWQ-W4A16", "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16-n64", 4),
    ("GPTQ-W4A16", "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16-n64", 4),
]

DEFAULT_DENSE_PARAMS_B = 8.767123696


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=PATH[:BITS]",
        help="Model entry. Can be repeated. Default: local Qwen3-VL quant variants.",
    )
    parser.add_argument(
        "--dense-params-b",
        type=float,
        default=DEFAULT_DENSE_PARAMS_B,
        help="Dense-equivalent parameter count in billions.",
    )
    parser.add_argument(
        "--generated-tokens",
        type=float,
        default=2.0,
        help="Generated tokens per call for FLOPs/call. Default matches current 8meta logs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="/mnt/2/carla_metric_result2/qwen3vl_quant_flops",
        help="Output prefix for .csv and .md.",
    )
    return parser.parse_args()


def parse_model_arg(arg: str) -> Tuple[str, str, Optional[int]]:
    if "=" not in arg:
        raise ValueError(f"--model must be LABEL=PATH[:BITS], got {arg!r}")
    label, rest = arg.split("=", 1)
    path = rest
    bits = None
    maybe_path, maybe_bits = rest.rsplit(":", 1) if ":" in rest else (rest, "")
    if maybe_bits.isdigit():
        path = maybe_path
        bits = int(maybe_bits)
    return label, path, bits


def weight_storage(path: Path) -> Tuple[int, float, float]:
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    total = 0
    files = 0
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix not in suffixes:
            continue
        if ".git" in item.parts:
            continue
        total += item.stat().st_size
        files += 1
    return files, total / 1e9, total / 1024**3


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def infer_bits(label: str, path: Path, override: Optional[int]) -> int:
    if override is not None:
        return override
    text = label.lower() + " " + path.name.lower()
    if "w8a8" in text or "int8" in text:
        return 8
    if "w4a16" in text or "gptq" in text or "awq" in text:
        return 4
    cfg = read_json(path / "carla_quantization_metadata.json")
    method = str(cfg.get("method", "")).lower()
    if "w8a8" in method or "int8" in method:
        return 8
    if "w4a16" in method or "gptq" in method or "awq" in method:
        return 4
    return 16


def build_rows(entries: List[Tuple[str, str, Optional[int]]], dense_params_b: float, generated_tokens: float) -> List[Dict[str, Any]]:
    rows = []
    dense_flops_per_token = 2.0 * dense_params_b * 1e9
    for label, path_str, bits_override in entries:
        path = Path(path_str)
        files, size_gb, size_gib = weight_storage(path)
        bits = infer_bits(label, path, bits_override)
        dense_gflops_token = dense_flops_per_token / 1e9
        dense_tflops_call = dense_flops_per_token * generated_tokens / 1e12
        bitops_token = dense_flops_per_token * bits
        rows.append(
            {
                "method": label,
                "model_path": str(path),
                "weight_files": files,
                "weight_size_GB": size_gb,
                "weight_size_GiB": size_gib,
                "dense_params_B": dense_params_b,
                "weight_bits_proxy": bits,
                "dense_equiv_GFLOPs_per_token": dense_gflops_token,
                "dense_equiv_TFLOPs_per_call": dense_tflops_call,
                "bitops_proxy_GbitOps_per_token": bitops_token / 1e9,
                "bitops_proxy_TbitOps_per_call": bitops_token * generated_tokens / 1e12,
                "generated_tokens_assumed": generated_tokens,
                "note": "Dense-equivalent FLOPs are same for same architecture; bitops proxy separates quant precision.",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        ("Method", "method", 0),
        ("Size(GB)", "weight_size_GB", 3),
        ("Params(B)", "dense_params_B", 3),
        ("Bits", "weight_bits_proxy", 0),
        ("GFLOPs/token", "dense_equiv_GFLOPs_per_token", 3),
        ("TFLOPs/call", "dense_equiv_TFLOPs_per_call", 6),
        ("GbitOps/token", "bitops_proxy_GbitOps_per_token", 3),
        ("TbitOps/call", "bitops_proxy_TbitOps_per_call", 6),
    ]
    lines = [
        "| " + " | ".join(c[0] for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for _, key, digits in cols:
            cells.append(fmt(row.get(key, ""), digits))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    entries = [parse_model_arg(x) for x in args.model] if args.model else DEFAULT_MODELS
    rows = build_rows(entries, args.dense_params_b, args.generated_tokens)
    prefix = Path(args.output_prefix)
    write_csv(Path(str(prefix) + ".csv"), rows)
    write_md(Path(str(prefix) + ".md"), rows)
    print(f"csv: {prefix}.csv")
    print(f"md:  {prefix}.md")


if __name__ == "__main__":
    main()
