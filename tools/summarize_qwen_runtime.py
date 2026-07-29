#!/usr/bin/env python3
"""Summarize Qwen inference cost logged during Bench2Drive scenarios.

Input can be a SAVE_PATH directory or a single qwen_intervention.jsonl file.
The script deduplicates repeated cached results and writes:
  <output-prefix>_calls.csv
  <output-prefix>_summary.csv
  <output-prefix>_paper_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BENCH_KEYS = ("qwen_benchmark", "rule_benchmark", "emergency_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="SAVE_PATH dir or qwen_intervention.jsonl")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output prefix. Default: <input>/qwen_runtime for dirs or beside file.",
    )
    parser.add_argument("--model-name", default="", help="Optional display model name for paper_table.csv")
    parser.add_argument("--quant", default="", help="Optional display quant name for paper_table.csv")
    return parser.parse_args()


def find_logs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("qwen_intervention.jsonl"))


def as_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def route_name(log_path: Path) -> str:
    return log_path.parent.name


def bench_get(bench: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Read current and legacy benchmark keys.

    Older 8meta logs used short names such as total_s/generation_s, while the
    paper-table summarizer expects explicit latency names.  Keeping the mapping
    here lets partially completed runs remain usable after a crash.
    """
    for key in keys:
        value = bench.get(key)
        if value not in (None, ""):
            return value
    return default


def nested_get(data: Dict[str, Any], path: str, default: Any = "") -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur in (None, ""):
            return default
    return cur


def iter_call_rows(log_path: Path) -> Iterable[Dict[str, Any]]:
    seen = set()
    with log_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in BENCH_KEYS:
                bench = entry.get(key)
                if not isinstance(bench, dict):
                    continue
                dedup_key = (
                    key,
                    bench.get("request_id"),
                    bench.get("request_step"),
                    bench.get("prompt_mode"),
                    bench.get("generated_tokens"),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                row = {
                    "route": route_name(log_path),
                    "log_path": str(log_path),
                    "line_no": line_no,
                    "bench_key": key,
                    "request_id": bench.get("request_id", ""),
                    "request_step": bench.get("request_step", ""),
                    "request_trigger": bench.get("request_trigger", ""),
                    "prompt_mode": bench_get(bench, "prompt_mode", default=entry.get("prompt_mode", "")),
                    "model": bench_get(bench, "model", "model_info.model"),
                    "model_type": bench_get(bench, "model_type", default=nested_get(bench, "model_info.model_type")),
                    "quant_method": bench_get(bench, "quant_method", default=bench_get(bench, "runtime_quant", default=entry.get("qwen_quant", ""))),
                    "quant_bits": bench_get(bench, "quant_bits"),
                    "runtime_quant": bench_get(bench, "runtime_quant", default=nested_get(bench, "model_info.runtime_quant")),
                    "runtime_quantized": bench_get(bench, "runtime_quantized", default=nested_get(bench, "model_info.runtime_quantized")),
                    "device": bench_get(bench, "device", default=bench_get(bench, "backend")),
                    "image_h": bench.get("image_h", ""),
                    "image_w": bench.get("image_w", ""),
                    "input_tokens": bench.get("input_tokens", ""),
                    "generated_tokens": bench.get("generated_tokens", ""),
                    "max_new_tokens": bench_get(bench, "max_new_tokens"),
                    "queue_wait_s": bench_get(bench, "queue_wait_s"),
                    "preprocess_latency_s": bench_get(bench, "preprocess_latency_s", "preprocess_s"),
                    "h2d_latency_s": bench_get(bench, "h2d_latency_s", "h2d_s"),
                    "generation_latency_s": bench_get(bench, "generation_latency_s", "generation_s"),
                    "decode_parse_latency_s": bench_get(bench, "decode_parse_latency_s", "decode_parse_s"),
                    "end_to_end_latency_s": bench_get(bench, "end_to_end_latency_s", "total_s"),
                    "tokens_per_s": bench_get(bench, "tokens_per_s"),
                    "param_count_billion": bench_get(bench, "param_count_billion", default=nested_get(bench, "model_info.param_count_billion")),
                    "param_storage_gib": bench_get(bench, "param_storage_gib", default=nested_get(bench, "model_info.param_storage_gib")),
                    "checkpoint_storage_gib": bench_get(bench, "checkpoint_storage_gib", default=nested_get(bench, "model_info.checkpoint_storage_gib")),
                    "avg_bits_per_param_storage": bench_get(bench, "avg_bits_per_param_storage"),
                    "load_time_s": bench_get(bench, "load_time_s", default=nested_get(bench, "load_memory.load_time_s")),
                    "load_memory_delta_allocated_gib": bench_get(bench, "load_memory_delta_allocated_gib", default=nested_get(bench, "load_memory.delta_allocated_gib")),
                    "load_memory_delta_reserved_gib": bench_get(bench, "load_memory_delta_reserved_gib", default=nested_get(bench, "load_memory.delta_reserved_gib")),
                    "load_memory_after_allocated_gib": bench_get(bench, "load_memory_after_allocated_gib", default=nested_get(bench, "load_memory.after_allocated_gib")),
                    "load_memory_after_reserved_gib": bench_get(bench, "load_memory_after_reserved_gib", default=nested_get(bench, "load_memory.after_reserved_gib")),
                    "approx_decode_tflops": bench.get("approx_decode_tflops", ""),
                    "approx_decode_tflops_per_s": bench.get("approx_decode_tflops_per_s", ""),
                    "cuda_allocated_gib": bench_get(bench, "cuda_allocated_gib"),
                    "cuda_reserved_gib": bench_get(bench, "cuda_reserved_gib"),
                    "cuda_max_allocated_gib": bench_get(bench, "cuda_max_allocated_gib", "peak_allocated_gib"),
                    "cuda_max_reserved_gib": bench_get(bench, "cuda_max_reserved_gib", "peak_reserved_gib"),
                    "generation_peak_delta_allocated_gib": bench.get("generation_peak_delta_allocated_gib", ""),
                    "generation_peak_delta_reserved_gib": bench.get("generation_peak_delta_reserved_gib", ""),
                    "cuda_free_gib": bench.get("cuda_free_gib", ""),
                    "cuda_total_gib": bench.get("cuda_total_gib", ""),
                }
                yield row


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - rank) + xs[hi] * (rank - lo)


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


def group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("model", ""),
        row.get("quant_method", ""),
        row.get("prompt_mode", ""),
        row.get("device", ""),
    )


def numeric(rows: List[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        val = as_float(row.get(key))
        if val is not None:
            vals.append(val)
    return vals


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)

    out: List[Dict[str, Any]] = []
    for (model, quant_method, prompt_mode, device), items in sorted(groups.items()):
        total = numeric(items, "end_to_end_latency_s")
        gen = numeric(items, "generation_latency_s")
        tps = numeric(items, "tokens_per_s")
        input_tokens = numeric(items, "input_tokens")
        new_tokens = numeric(items, "generated_tokens")
        peak_mem = numeric(items, "cuda_max_allocated_gib")
        peak_delta = numeric(items, "generation_peak_delta_allocated_gib")
        load_delta = numeric(items, "load_memory_delta_allocated_gib")
        load_after = numeric(items, "load_memory_after_allocated_gib")
        tflops = numeric(items, "approx_decode_tflops")
        out.append(
            {
                "model": model,
                "quant_method": quant_method,
                "prompt_mode": prompt_mode,
                "device": device,
                "calls": len(items),
                "end_to_end_mean_s": statistics.mean(total) if total else "",
                "end_to_end_median_s": statistics.median(total) if total else "",
                "end_to_end_p95_s": percentile(total, 95) if total else "",
                "generation_mean_s": statistics.mean(gen) if gen else "",
                "generation_median_s": statistics.median(gen) if gen else "",
                "generation_p95_s": percentile(gen, 95) if gen else "",
                "tokens_per_s_mean": statistics.mean(tps) if tps else "",
                "input_tokens_mean": statistics.mean(input_tokens) if input_tokens else "",
                "generated_tokens_mean": statistics.mean(new_tokens) if new_tokens else "",
                "load_memory_delta_allocated_gib": max(load_delta) if load_delta else "",
                "load_memory_after_allocated_gib": max(load_after) if load_after else "",
                "cuda_max_allocated_gib_max": max(peak_mem) if peak_mem else "",
                "generation_peak_delta_allocated_gib_max": max(peak_delta) if peak_delta else "",
                "approx_decode_tflops_mean": statistics.mean(tflops) if tflops else "",
                "param_count_billion": items[0].get("param_count_billion", ""),
                "param_storage_gib": items[0].get("param_storage_gib", ""),
                "checkpoint_storage_gib": items[0].get("checkpoint_storage_gib", ""),
                "avg_bits_per_param_storage": items[0].get("avg_bits_per_param_storage", ""),
            }
        )
    return out


def paper_table(rows: List[Dict[str, Any]], model_name: str = "", quant_name: str = "") -> List[Dict[str, Any]]:
    if not rows:
        return []
    total = numeric(rows, "end_to_end_latency_s")
    tps = numeric(rows, "tokens_per_s")
    peak_mem = numeric(rows, "cuda_max_allocated_gib")
    peak_delta = numeric(rows, "generation_peak_delta_allocated_gib")
    load_delta = numeric(rows, "load_memory_delta_allocated_gib")
    load_after = numeric(rows, "load_memory_after_allocated_gib")
    peak_delta_reserved = numeric(rows, "generation_peak_delta_reserved_gib")
    load_delta_reserved = numeric(rows, "load_memory_delta_reserved_gib")
    load_after_reserved = numeric(rows, "load_memory_after_reserved_gib")
    first = rows[0]
    display_model = model_name or _short_model_name(str(first.get("model", "")))
    display_quant = quant_name or str(first.get("quant_method", "") or "unknown")
    if str(first.get("quant_method", "")).lower() == "fp8" and str(first.get("runtime_quantized", "")).lower() in {"false", "0"}:
        display_quant = f"{display_quant} (dequantized)"
    load_memory = (
        max(load_delta)
        if load_delta
        else (
            max(load_delta_reserved)
            if load_delta_reserved
            else (
                max(load_after)
                if load_after
                else (max(load_after_reserved) if load_after_reserved else "")
            )
        )
    )
    peak_memory = (
        max(load_delta) + max(peak_delta)
        if load_delta and peak_delta
        else (
            max(load_delta_reserved) + max(peak_delta_reserved)
            if load_delta_reserved and peak_delta_reserved
            else (max(peak_mem) if peak_mem else load_memory)
        )
    )
    model_size = _checkpoint_weight_storage_gib(str(first.get("model", "")))
    if model_size is None:
        model_size = first.get("checkpoint_storage_gib", "") or first.get("param_storage_gib", "")
    return [
        {
            "Model": display_model,
            "Quant": display_quant,
            "Params(B)": _round_value(first.get("param_count_billion", ""), 3),
            "Model Size(GB) ↓": _round_value(model_size, 3),
            "Load Memory(GB) ↓": _round_value(load_memory, 3),
            "Peak Memory(GB) ↓": _round_value(peak_memory, 3),
            "Avg Latency(ms) ↓": _round_value(statistics.mean(total) * 1000.0 if total else "", 2),
            "P95 Latency(ms) ↓": _round_value(percentile(total, 95) * 1000.0 if total else "", 2),
            "Tokens/sec ↑": _round_value(statistics.mean(tps) if tps else "", 3),
            "Calls": len(rows),
        }
    ]


def _short_model_name(model: str) -> str:
    if not model:
        return ""
    return Path(model.rstrip("/")).name or model


def _checkpoint_weight_storage_gib(model: str) -> Optional[float]:
    if not model:
        return None
    root = Path(model)
    if not root.exists() or not root.is_dir():
        return None
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    total = 0
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if ".git" in path.parts:
                continue
            total += path.stat().st_size
    except OSError:
        return None
    return total / 1024**3 if total else None


def _round_value(value: Any, digits: int):
    val = as_float(value)
    return round(val, digits) if val is not None else value


def default_prefix(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name("qwen_runtime")
    return input_path / "qwen_runtime"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    prefix = Path(args.output_prefix) if args.output_prefix else default_prefix(input_path)

    rows: List[Dict[str, Any]] = []
    for log_path in find_logs(input_path):
        rows.extend(iter_call_rows(log_path))

    calls_path = Path(str(prefix) + "_calls.csv")
    summary_path = Path(str(prefix) + "_summary.csv")
    paper_path = Path(str(prefix) + "_paper_table.csv")
    write_csv(calls_path, rows)
    write_csv(summary_path, summarize(rows))
    write_csv(paper_path, paper_table(rows, model_name=args.model_name, quant_name=args.quant))
    print(f"calls:   {calls_path} ({len(rows)} calls)")
    print(f"summary: {summary_path}")
    print(f"paper:   {paper_path}")


if __name__ == "__main__":
    main()
