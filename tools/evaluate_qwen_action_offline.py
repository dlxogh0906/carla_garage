#!/usr/bin/env python3
"""Offline action-id evaluation for Qwen3-VL meta-action ablation models."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


META_ACTIONS = [
    ("proceed", 1.0),
    ("slow_down", 0.6),
    ("stop", 0.0),
    ("yield", 0.3),
    ("turn_left", 0.7),
    ("turn_right", 0.7),
    ("change_lane_left", 0.8),
    ("change_lane_right", 0.8),
]

ACTION_LIST = "\n".join(f"  {i}: {name} (speed x{mult})" for i, (name, mult) in enumerate(META_ACTIONS))
PROMPT = (
    "You are a driving safety assistant for an autonomous vehicle.\n"
    "Analyze the front camera image and select the single best meta-action.\n\n"
    "Meta-actions:\n"
    f"{ACTION_LIST}\n\n"
    "Reply with exactly one character from 0,1,2,3,4,5,6,7.\n"
    "Do not output punctuation, words, markdown, or explanations.\n"
    "/no_think"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--backend", choices=("transformers", "openai"), default="transformers")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_action(text: str) -> int | None:
    stripped = text.strip()
    for i in range(8):
        if stripped.startswith(str(i)):
            return i
    match = re.search(r"(?<!\d)([0-7])(?!\d)", stripped)
    if match:
        return int(match.group(1))
    lower = text.lower()
    for i, (name, _mult) in enumerate(META_ACTIONS):
        if name in lower or name.replace("_", " ") in lower:
            return i
    return None


def load_transformers(model_name: str, device: str, dtype_name: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return processor, model, dtype


def infer_transformers(processor, model, dtype, image_path: Path, max_new_tokens: int) -> tuple[str, int, int, float]:
    import torch

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_device = next(model.parameters()).device
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model_device)
    for key, value in list(inputs.items()):
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            inputs[key] = value.to(dtype=dtype)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    n_in = int(inputs["input_ids"].shape[1])
    response = processor.decode(out[0][n_in:], skip_special_tokens=True).strip()
    return response, n_in, int(out.shape[1] - n_in), elapsed_ms


def image_data_url(path: Path) -> str:
    import base64

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def infer_openai(endpoint: str, model: str, image_path: Path, max_new_tokens: int) -> tuple[str, int, int, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    start = time.perf_counter()
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    parsed = json.loads(body)
    response = str(parsed.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    usage = parsed.get("usage") or {}
    return response, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or len(response)), elapsed_ms


def main() -> None:
    args = parse_args()
    val_path = Path(args.validation_jsonl).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(val_path)
    if args.limit:
        rows = rows[: args.limit]

    processor = model = dtype = None
    if args.backend == "transformers":
        processor, model, dtype = load_transformers(args.model, args.device, args.dtype)

    correct = 0
    invalid = 0
    latencies = []
    outputs = []
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            image_path = Path(row["image"]).expanduser().resolve()
            if args.backend == "transformers":
                response, prompt_tokens, generated_tokens, latency_ms = infer_transformers(
                    processor, model, dtype, image_path, args.max_new_tokens
                )
            else:
                response, prompt_tokens, generated_tokens, latency_ms = infer_openai(
                    args.endpoint, args.model, image_path, args.max_new_tokens
                )
            pred = parse_action(response)
            label = int(row["label"])
            is_invalid = pred is None
            is_correct = (pred == label) if pred is not None else False
            correct += int(is_correct)
            invalid += int(is_invalid)
            latencies.append(latency_ms)
            out = {
                **row,
                "model": args.model,
                "pred": pred,
                "correct": is_correct,
                "invalid": is_invalid,
                "response": response,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "latency_ms": latency_ms,
            }
            outputs.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(rows)}] label={label} pred={pred} ok={is_correct} invalid={is_invalid} latency_ms={latency_ms:.1f} raw={response!r}")

    n = len(rows)
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(0.95 * (n - 1))] if n else 0.0
    summary = {
        "validation_jsonl": str(val_path),
        "model": args.model,
        "backend": args.backend,
        "num_samples": n,
        "accuracy": correct / n if n else 0.0,
        "invalid_rate": invalid / n if n else 0.0,
        "avg_latency_ms": sum(latencies) / n if n else 0.0,
        "p95_latency_ms": p95,
        "correct": correct,
        "invalid": invalid,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
