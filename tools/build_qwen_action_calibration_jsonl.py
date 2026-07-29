#!/usr/bin/env python3
"""Build Qwen3-VL calibration JSONL from offline meta-action image rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor

from evaluate_qwen_action_offline import PROMPT


DEFAULT_MODEL = "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Rows with an image or image_path field.")
    parser.add_argument("--output-jsonl", required=True, help="Calibration JSONL to write.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model path used to render Qwen chat_template_text.",
    )
    parser.add_argument("--limit", type=int, help="Optional row limit.")
    parser.add_argument("--image-field", default="image", help="Input image field. Falls back to image_path.")
    parser.add_argument("--prompt-mode", default="team8_meta_action_digit")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No input rows found in {path}")
    return rows


def image_path_from_row(row: dict[str, Any], image_field: str) -> Path:
    raw_path = row.get(image_field) or row.get("image_path")
    if not raw_path:
        raise KeyError(f"Row has neither {image_field!r} nor 'image_path': {row}")
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_chat_template_text(processor: Any, image_path: Path) -> str:
    with Image.open(image_path) as src:
        image = src.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def calibration_row(
    processor: Any,
    source_path: Path,
    idx: int,
    row: dict[str, Any],
    image_field: str,
    prompt_mode: str,
) -> dict[str, Any]:
    image_path = image_path_from_row(row, image_field)
    route = str(row.get("route") or "row")
    step = row.get("image_step", row.get("step", idx))
    return {
        "sample_id": f"action_calib_{idx:05d}_{route}_step{step}",
        "image_path": str(image_path),
        "prompt_text": PROMPT,
        "chat_template_text": build_chat_template_text(processor, image_path),
        "prompt_mode": prompt_mode,
        "source_jsonl": str(source_path),
        "source_row_index": idx,
        "source_label": row.get("label"),
        "source_label_name": row.get("label_name", ""),
    }


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_jsonl, args.limit)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            item = calibration_row(
                processor=processor,
                source_path=input_jsonl,
                idx=idx,
                row=row,
                image_field=args.image_field,
                prompt_mode=args.prompt_mode,
            )
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(rows)}] {item['sample_id']} {item['image_path']}")

    print(f"wrote calibration rows: {len(rows)}")
    print(f"output: {output_jsonl}")


if __name__ == "__main__":
    main()
