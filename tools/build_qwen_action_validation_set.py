#!/usr/bin/env python3
"""Build a small Qwen meta-action validation JSONL from CARLA logs and ClassicCV images.

The current ClassicCV runs store side-by-side comparison images
(left=original, right=ClassicCV-enhanced).  For offline VLM validation this
script crops the right half so that the validation image matches the enhanced
image path used by the agent as closely as the saved artifacts allow.

Labels are pseudo-labels from the existing qwen_intervention.jsonl action_idx.
Use them for model-to-model consistency/ablation screening, then confirm the
top candidates with CARLA dev10.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


STEP_RE = re.compile(r"step_(\d+)\.jpg$")
ROUTE_RE = re.compile(r"(RouteScenario_\d+_rep\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="CARLA metric run directory.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-step-delta", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dedupe-request",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate cached log rows that refer to the same route/request/image/label.",
    )
    parser.add_argument(
        "--only-vlm-called",
        action="store_true",
        help="Keep rows where the log says a VLM request/result happened.",
    )
    return parser.parse_args()


def route_id_from_path(path: Path) -> str | None:
    for part in path.parts:
        match = ROUTE_RE.search(part)
        if match:
            return match.group(1)
    return None


def step_from_image(path: Path) -> int | None:
    match = STEP_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_compare_images(run_dir: Path) -> dict[str, dict[int, Path]]:
    images: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted((run_dir / "classiccv_compare").glob("**/step_*.jpg")):
        route = route_id_from_path(path)
        step = step_from_image(path)
        if route is None or step is None:
            continue
        images[route][step] = path
    return images


def crop_enhanced_right_half(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        width, height = img.size
        crop = img.crop((width // 2, 0, width, height)).convert("RGB")
        crop.save(dst, quality=95)


def is_vlm_related(row: dict[str, Any]) -> bool:
    if row.get("vlm_called") is True:
        return True
    if row.get("qwen_request_step") is not None:
        return True
    if str(row.get("qwen_raw_response") or "").strip():
        return True
    return False


def choose_image(route_images: dict[int, Path], target_step: int, max_delta: int) -> tuple[int, Path] | None:
    if not route_images:
        return None
    best_step = min(route_images, key=lambda s: (abs(s - target_step), s))
    if abs(best_step - target_step) > max_delta:
        return None
    return best_step, route_images[best_step]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    image_output_dir = Path(args.image_output_dir).expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    compare_images = collect_compare_images(run_dir)
    candidates: list[dict[str, Any]] = []
    seen_requests: set[tuple[str, int, int, int]] = set()
    for log_path in sorted(run_dir.glob("**/qwen_intervention.jsonl")):
        route = route_id_from_path(log_path)
        if route is None or route not in compare_images:
            continue
        for row in read_jsonl(log_path):
            if args.only_vlm_called and not is_vlm_related(row):
                continue
            if "action_idx" not in row:
                continue
            try:
                label = int(row["action_idx"])
            except (TypeError, ValueError):
                continue
            if not 0 <= label <= 7:
                continue
            target_step = row.get("qwen_request_step")
            if target_step is None:
                target_step = row.get("step")
            if target_step is None:
                continue
            target_step = int(target_step)
            chosen = choose_image(compare_images[route], target_step, args.max_step_delta)
            if chosen is None:
                continue
            image_step, compare_path = chosen
            dedupe_key = (route, target_step, image_step, label)
            if args.dedupe_request and dedupe_key in seen_requests:
                continue
            seen_requests.add(dedupe_key)
            candidates.append(
                {
                    "route": route,
                    "step": int(row.get("step", target_step)),
                    "target_step": target_step,
                    "image_step": image_step,
                    "step_delta": abs(image_step - target_step),
                    "label": label,
                    "label_name": row.get("action_name", ""),
                    "source_compare_image": str(compare_path),
                    "qwen_raw_response": row.get("qwen_raw_response", ""),
                    "vlm_called": bool(row.get("vlm_called", False)),
                    "ttc": row.get("ttc"),
                    "is_risky": row.get("is_risky"),
                    "prompt_mode": row.get("prompt_mode", "team8_meta_action_digit"),
                    "source_log": str(log_path),
                }
            )

    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.max_samples]
    with output_jsonl.open("w", encoding="utf-8") as f:
        for i, item in enumerate(selected):
            image_name = f"{i:05d}_{item['route']}_step{item['image_step']:05d}.jpg"
            dst = image_output_dir / image_name
            crop_enhanced_right_half(Path(item["source_compare_image"]), dst)
            item["image"] = str(dst)
            item["image_source_note"] = "right_half_of_classiccv_compare_image"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    label_counts: dict[int, int] = defaultdict(int)
    for item in selected:
        label_counts[int(item["label"])] += 1
    print(f"candidates: {len(candidates)}")
    print(f"selected  : {len(selected)}")
    print(f"output    : {output_jsonl}")
    print(f"images    : {image_output_dir}")
    print(f"labels    : {dict(sorted(label_counts.items()))}")
    if not args.only_vlm_called:
        print("note      : labels are pseudo-labels from logged action_idx, not ground-truth annotations")


if __name__ == "__main__":
    main()
