#!/usr/bin/env python3
"""Summarize approximate Qwen decode FLOPs per model/run.

This uses the standard inference proxy:
  decode FLOPs ~= 2 * parameter_count * generated_tokens

For quantized checkpoints, the theoretical operation count is usually similar;
the speedup comes from lower precision kernels and memory bandwidth.  Therefore
this script reports both FLOPs per token/call and effective TFLOPs/s from the
measured latency.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True, help="Suite directory with run subdirectories.")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output prefix. Default: <suite-root>/qwen_flops",
    )
    return parser.parse_args()


def as_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def first_float(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    for row in rows:
        val = as_float(row.get(key))
        if val is not None:
            return val
    return None


def run_rows(suite_root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in suite_root.iterdir() if p.is_dir()):
        calls = read_csv(run_dir / "qwen_runtime_calls.csv")
        if not calls:
            continue
        params_b = first_float(calls, "param_count_billion")
        if params_b is None:
            continue
        gen_tokens = [x for x in (as_float(r.get("generated_tokens")) for r in calls) if x is not None]
        input_tokens = [x for x in (as_float(r.get("input_tokens")) for r in calls) if x is not None]
        gen_latency = [x for x in (as_float(r.get("generation_latency_s")) for r in calls) if x is not None]
        e2e_latency = [x for x in (as_float(r.get("end_to_end_latency_s")) for r in calls) if x is not None]
        logged_tflops = [x for x in (as_float(r.get("approx_decode_tflops")) for r in calls) if x is not None]
        logged_tflops_s = [
            x for x in (as_float(r.get("approx_decode_tflops_per_s")) for r in calls) if x is not None
        ]

        flops_per_token = 2.0 * params_b * 1e9
        gflops_per_token = flops_per_token / 1e9
        avg_gen_tokens = mean(gen_tokens)
        tflops_per_call = (
            (flops_per_token * avg_gen_tokens) / 1e12
            if avg_gen_tokens is not None
            else mean(logged_tflops)
        )
        avg_gen_latency = mean(gen_latency)
        effective_tflops_s = (
            tflops_per_call / avg_gen_latency
            if tflops_per_call is not None and avg_gen_latency and avg_gen_latency > 0
            else mean(logged_tflops_s)
        )

        first = calls[0]
        out.append(
            {
                "run_id": run_dir.name,
                "model": first.get("model", ""),
                "quant": first.get("quant_method", run_dir.name),
                "backend_device": first.get("device", ""),
                "calls": len(calls),
                "params_b": params_b,
                "avg_input_tokens": mean(input_tokens),
                "avg_generated_tokens": avg_gen_tokens,
                "decode_gflops_per_token": gflops_per_token,
                "decode_tflops_per_call": tflops_per_call,
                "effective_decode_tflops_per_s": effective_tflops_s,
                "avg_generation_latency_ms": avg_gen_latency * 1000.0 if avg_gen_latency is not None else None,
                "avg_e2e_latency_ms": mean(e2e_latency) * 1000.0 if e2e_latency else None,
                "note": "approx_decode_flops=2*params*generated_tokens",
            }
        )
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    x = as_float(value)
    return f"{x:.{digits}f}" if x is not None else str(value)


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


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        ("Run", "run_id", 0),
        ("Quant", "quant", 0),
        ("Params(B)", "params_b", 3),
        ("In Tok", "avg_input_tokens", 1),
        ("Out Tok", "avg_generated_tokens", 1),
        ("GFLOPs/token", "decode_gflops_per_token", 3),
        ("TFLOPs/call", "decode_tflops_per_call", 6),
        ("Effective TFLOPs/s", "effective_decode_tflops_per_s", 3),
        ("Gen Lat(ms)", "avg_generation_latency_ms", 2),
    ]
    lines = [
        "| " + " | ".join(c[0] for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells = []
        for _, key, digits in cols:
            val = row.get(key)
            cells.append(fmt(val, digits) if isinstance(val, (int, float)) or val is None else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root)
    prefix = Path(args.output_prefix) if args.output_prefix else suite_root / "qwen_flops"
    rows = run_rows(suite_root)
    write_csv(Path(str(prefix) + ".csv"), rows)
    write_md(Path(str(prefix) + ".md"), rows)
    print(f"csv: {prefix}.csv")
    print(f"md:  {prefix}.md")


if __name__ == "__main__":
    main()
