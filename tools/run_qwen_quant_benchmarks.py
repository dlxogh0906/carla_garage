#!/usr/bin/env python3
"""Run Qwen-VL quantization benchmarks for multiple model paths and write CSV.

Each model can be passed as either:
  label=/path/to/model
or just:
  /path/to/model

For AWQ/GPTQ/FP8 checkpoints, this runner uses benchmark_qwen_vl.py with
--quant auto so the checkpoint's own quantization_config is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_OUT = "/mnt/2/carla_metric_result/qwen_benchmarks/quant_suite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model paths or label=path entries. First successful model is the baseline.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image", default="")
    parser.add_argument("--image-size", type=int, nargs=2, default=[900, 1600], metavar=("H", "W"))
    parser.add_argument("--prompt", default="Describe the driving scene and answer with one short JSON object.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-visible-devices", default="", help="Optional CUDA_VISIBLE_DEVICES override")
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--max-memory", default="")
    parser.add_argument("--offload-folder", default="")
    parser.add_argument(
        "--runtime-quant",
        choices=["auto", "none", "bnb4", "bnb8"],
        default="auto",
        help="Use auto for pre-quantized AWQ/GPTQ/FP8 checkpoints.",
    )
    return parser.parse_args()


def split_model_item(item: str) -> Tuple[str, str]:
    if "=" in item and not item.startswith("/"):
        label, path = item.split("=", 1)
        return label.strip(), path.strip()
    path = item.strip()
    return infer_label(path), path


def infer_label(path: str) -> str:
    name = Path(path).name or re.sub(r"[^A-Za-z0-9_.-]+", "_", path.strip("/"))
    low = name.lower()
    if "awq" in low:
        prefix = "awq"
    elif "gptq" in low:
        prefix = "gptq"
    elif "fp8" in low:
        prefix = "fp8"
    elif "int4" in low or "w4" in low:
        prefix = "int4"
    elif "int8" in low:
        prefix = "int8"
    else:
        prefix = "fp16"
    return f"{prefix}_{slug(name)}"


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("._")[:120] or "model"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_nested(d: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def row_from_report(label: str, report: Optional[Dict[str, Any]], status: str, error: str) -> Dict[str, Any]:
    report = report or {}
    image_hw = report.get("image_size_hw") or ["", ""]
    quant = report.get("quantization") or {}
    params = report.get("params") or {}
    mem = report.get("cuda_memory") or {}
    latency = report.get("latency") or {}
    flops = report.get("flops_proxy") or {}
    return {
        "label": label,
        "model": report.get("model", ""),
        "status": status,
        "error": error,
        "model_type": report.get("model_type", ""),
        "model_class": report.get("model_class", ""),
        "requested_quant": quant.get("requested_quant", report.get("quant", "")),
        "detected_quant_method": quant.get("detected_method", ""),
        "quant_bits": quant.get("bits", ""),
        "dtype": report.get("dtype", ""),
        "device": report.get("device", ""),
        "total_params_b": params.get("total_params_billion", ""),
        "param_storage_gib": params.get("param_storage_gib", ""),
        "avg_bits_per_param_storage": params.get("avg_bits_per_param_from_storage", ""),
        "load_time_s": report.get("load_time_s", ""),
        "input_tokens": report.get("input_tokens", ""),
        "image_h": image_hw[0],
        "image_w": image_hw[1],
        "max_new_tokens": report.get("max_new_tokens", ""),
        "warmup": report.get("warmup", ""),
        "repeat": report.get("repeat", ""),
        "latency_mean_s": latency.get("latency_s_mean", ""),
        "latency_median_s": latency.get("latency_s_median", ""),
        "latency_p95_s": latency.get("latency_s_p95", ""),
        "latency_min_s": latency.get("latency_s_min", ""),
        "latency_max_s": latency.get("latency_s_max", ""),
        "tokens_per_s_mean": latency.get("tokens_per_s_mean", ""),
        "tokens_per_s_median": latency.get("tokens_per_s_median", ""),
        "new_tokens_mean": latency.get("new_tokens_mean", ""),
        "cuda_max_allocated_gib": mem.get("max_allocated_gib", ""),
        "cuda_max_reserved_gib": mem.get("max_reserved_gib", ""),
        "cuda_allocated_gib": mem.get("allocated_gib", ""),
        "cuda_reserved_gib": mem.get("reserved_gib", ""),
        "approx_decode_tflops_mean": flops.get("approx_decode_tflops_for_mean_new_tokens", ""),
        "approx_decode_flops_per_token": flops.get("approx_decode_flops_per_generated_token", ""),
        "latency_speedup_vs_baseline": "",
        "tokens_per_s_ratio_vs_baseline": "",
        "peak_mem_reduction_pct_vs_baseline": "",
        "param_storage_reduction_pct_vs_baseline": "",
    }


def as_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def add_baseline_columns(rows: List[Dict[str, Any]]) -> None:
    baseline = next((r for r in rows if r.get("status") == "ok"), None)
    if baseline is None:
        return
    base_lat = as_float(baseline.get("latency_mean_s"))
    base_tps = as_float(baseline.get("tokens_per_s_mean"))
    base_mem = as_float(baseline.get("cuda_max_allocated_gib"))
    base_storage = as_float(baseline.get("param_storage_gib"))

    for row in rows:
        lat = as_float(row.get("latency_mean_s"))
        tps = as_float(row.get("tokens_per_s_mean"))
        mem = as_float(row.get("cuda_max_allocated_gib"))
        storage = as_float(row.get("param_storage_gib"))
        if base_lat and lat:
            row["latency_speedup_vs_baseline"] = base_lat / lat
        if base_tps and tps:
            row["tokens_per_s_ratio_vs_baseline"] = tps / base_tps
        if base_mem and mem is not None:
            row["peak_mem_reduction_pct_vs_baseline"] = 100.0 * (base_mem - mem) / base_mem
        if base_storage and storage is not None:
            row["param_storage_reduction_pct_vs_baseline"] = 100.0 * (base_storage - storage) / base_storage


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).with_name("benchmark_qwen_vl.py")
    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(args.models):
        label, model = split_model_item(item)
        out_json = output_dir / f"{idx:02d}_{slug(label)}.json"
        out_stdout = output_dir / f"{idx:02d}_{slug(label)}.stdout.txt"
        out_stderr = output_dir / f"{idx:02d}_{slug(label)}.stderr.txt"

        cmd = [
            args.python,
            str(script),
            "--model",
            model,
            "--label",
            label,
            "--device",
            args.device,
            "--quant",
            args.runtime_quant,
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--prompt",
            args.prompt,
            "--output",
            str(out_json),
        ]
        if args.image:
            cmd += ["--image", args.image]
        else:
            cmd += ["--image-size", str(args.image_size[0]), str(args.image_size[1])]
        if args.profile_flops:
            cmd.append("--profile-flops")
        if args.device_map:
            cmd += ["--device-map", args.device_map]
        if args.max_memory:
            cmd += ["--max-memory", args.max_memory]
        if args.offload_folder:
            cmd += ["--offload-folder", str(Path(args.offload_folder) / slug(label))]

        env = os.environ.copy()
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        print(f"\n[quant-bench] {label}: {model}")
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        out_stdout.write_text(proc.stdout)
        out_stderr.write_text(proc.stderr)
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        report = read_json(out_json)
        status = "ok" if proc.returncode == 0 and report is not None else "failed"
        error = "" if status == "ok" else (proc.stderr or proc.stdout)[-1000:]
        rows.append(row_from_report(label, report, status, error))

    add_baseline_columns(rows)
    write_csv(output_dir / "summary.csv", rows)
    print(f"\n[quant-bench] wrote {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
