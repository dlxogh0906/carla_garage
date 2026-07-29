#!/usr/bin/env python3
"""Create a mixed-bit GPTQ Qwen3-VL checkpoint with llm-compressor.

Default policy:
  - keep visual encoder blocks and lm_head in BF16/FP16
  - quantize visual merger/projection layers to W8A16
  - quantize first/last language layers to W8A16
  - quantize middle language layers to W4A16 GPTQ
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoConfig, AutoProcessor

from quantize_qwen3vl_w4a16 import (
    DEFAULT_CALIB,
    DEFAULT_MODEL,
    dtype_from_name,
    load_calibration_dataset,
    parse_max_memory,
    pick_model_class,
    qwen_vl_batch1_collator,
)


DEFAULT_OUT = "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-MixedGPTQ-W4W8A16"
TEXT_LINEAR_SUFFIX_RE = (
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))"
)
HF_PROJECTION_RE = r"model\.visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]"
VLLM_PROJECTION_RE = r"visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--calib-jsonl", default=DEFAULT_CALIB)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-memory",
        default="",
        help="Optional accelerate max_memory, e.g. '0=36GiB,1=36GiB,cpu=240GiB'.",
    )
    parser.add_argument("--offload-folder", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dry-plan", action="store_true", help="Print bit allocation and exit.")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--actorder", default="static", choices=["static", "group", "weight", "none"])
    parser.add_argument("--offload-hessians", action="store_true")
    parser.add_argument(
        "--w8-first-layers",
        type=int,
        default=2,
        help="Number of earliest language decoder layers kept at W8A16.",
    )
    parser.add_argument(
        "--w8-last-layers",
        type=int,
        default=2,
        help="Number of latest language decoder layers kept at W8A16.",
    )
    parser.add_argument(
        "--w8-layers",
        default="",
        help="Explicit comma-separated language layer ids for W8A16. Overrides first/last policy.",
    )
    parser.add_argument(
        "--no-w8-projection",
        action="store_true",
        help="Leave visual merger/projection in BF16 instead of W8A16.",
    )
    parser.add_argument(
        "--quantize-visual-blocks",
        action="store_true",
        help="Also include visual encoder block Linear layers in W8A16. Default keeps them BF16.",
    )
    return parser.parse_args()


def get_num_text_layers(model_path: str) -> int:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None)
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None:
        raise RuntimeError("Could not infer number of text layers from config.")
    return int(num_layers)


def layer_regexes(layer_ids: list[int]) -> list[str]:
    if not layer_ids:
        return []
    layer_alt = "|".join(str(i) for i in layer_ids)
    return [
        rf"re:model\.language_model\.layers\.(?:{layer_alt})\.{TEXT_LINEAR_SUFFIX_RE}$",
        rf"re:language_model\.model\.layers\.(?:{layer_alt})\.{TEXT_LINEAR_SUFFIX_RE}$",
    ]


def build_mixed_scheme(
    num_layers: int,
    w8_first_layers: int,
    w8_last_layers: int,
    explicit_w8_layers: list[int] | None,
    include_projection: bool,
    include_visual_blocks: bool,
) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    if explicit_w8_layers is not None:
        w8_layers = sorted({idx for idx in explicit_w8_layers if 0 <= idx < num_layers})
    else:
        first = set(range(max(0, min(w8_first_layers, num_layers))))
        last_start = max(0, num_layers - max(0, w8_last_layers))
        last = set(range(last_start, num_layers))
        w8_layers = sorted(first | last)
    w4_layers = [idx for idx in range(num_layers) if idx not in set(w8_layers)]

    w8_targets: list[str] = []
    w4_targets: list[str] = []
    w8_targets.extend(layer_regexes(w8_layers))
    w4_targets.extend(layer_regexes(w4_layers))
    if include_projection:
        w8_targets.extend([f"re:{HF_PROJECTION_RE}$", f"re:{VLLM_PROJECTION_RE}$"])
    if include_visual_blocks:
        w8_targets.extend(
            [
                r"re:model\.visual\.blocks\.\d+\.attn\.(?:qkv|proj)$",
                r"re:model\.visual\.blocks\.\d+\.mlp\.linear_fc[12]$",
            ]
        )

    scheme: dict[str, list[str]] = {}
    if w4_targets:
        scheme["W4A16"] = w4_targets
    if w8_targets:
        scheme["W8A16"] = w8_targets

    ignore = ["lm_head"]
    if not include_visual_blocks:
        ignore.append(r"re:model\.visual\.blocks\..*")
    if not include_projection:
        ignore.append(r"re:model\.visual\.(?:merger|deepstack_merger_list\.\d+)\..*")

    plan = {
        "num_text_layers": num_layers,
        "w8_text_layers": w8_layers,
        "w4_text_layers": w4_layers,
        "projection": "W8A16" if include_projection else "BF16",
        "visual_blocks": "W8A16" if include_visual_blocks else "BF16",
        "lm_head": "BF16",
    }
    return scheme, ignore, plan


def parse_layer_ids(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    layer_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        layer_ids.append(int(item))
    return layer_ids


def infer_modules_from_index(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    data = json.loads(index_path.read_text())
    modules: list[str] = []
    for key in data.get("weight_map", {}):
        if key.endswith(".weight"):
            modules.append(key.removesuffix(".weight"))
    return modules


def count_regex_matches(modules: list[str], targets: list[str]) -> int:
    count = 0
    for name in modules:
        for target in targets:
            if target.startswith("re:") and re.match(target.removeprefix("re:"), name):
                count += 1
                break
            if target == name:
                count += 1
                break
    return count


def print_plan(model_path: Path, scheme: dict[str, list[str]], ignore: list[str], plan: dict[str, Any]) -> None:
    print("========================================")
    print("Qwen3-VL Mixed GPTQ bit allocation plan")
    print(f"model        : {model_path}")
    print(f"text layers  : {plan['num_text_layers']}")
    print(f"W8 text      : {plan['w8_text_layers']}")
    print(f"W4 text      : {plan['w4_text_layers'][0]}..{plan['w4_text_layers'][-1]}" if plan["w4_text_layers"] else "W4 text      : []")
    print(f"projection   : {plan['projection']}")
    print(f"visual blocks: {plan['visual_blocks']}")
    print(f"lm_head      : {plan['lm_head']}")
    print(f"ignore       : {ignore}")
    modules = infer_modules_from_index(model_path)
    if modules:
        for bits, targets in scheme.items():
            print(f"{bits} matches : {count_regex_matches(modules, targets)} modules")
    print("scheme:")
    print(json.dumps(scheme, indent=2))
    print("========================================")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    calib_path = Path(args.calib_jsonl)
    output_dir = Path(args.output_dir)
    num_layers = get_num_text_layers(str(model_path))
    explicit_w8_layers = parse_layer_ids(args.w8_layers)
    scheme, ignore, plan = build_mixed_scheme(
        num_layers=num_layers,
        w8_first_layers=args.w8_first_layers,
        w8_last_layers=args.w8_last_layers,
        explicit_w8_layers=explicit_w8_layers,
        include_projection=not args.no_w8_projection,
        include_visual_blocks=args.quantize_visual_blocks,
    )

    print_plan(model_path, scheme, ignore, plan)
    if args.dry_plan:
        return

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.no_save:
        raise RuntimeError(f"Output dir already exists and is not empty: {output_dir}")
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_calibration_dataset(calib_path, args.num_samples)
    num_samples = min(args.num_samples, len(dataset))

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, use_fast=True)
    _cfg, model_cls, class_name = pick_model_class(str(model_path))
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

    actorder = None if args.actorder == "none" else args.actorder
    recipe = [
        GPTQModifier(
            scheme=scheme,
            ignore=ignore,
            block_size=args.block_size,
            dampening_frac=args.dampening_frac,
            actorder=actorder,
            offload_hessians=args.offload_hessians,
        )
    ]

    oneshot(
        model=model,
        processor=processor,
        dataset=dataset,
        recipe=recipe,
        batch_size=1,
        data_collator=qwen_vl_batch1_collator,
        num_calibration_samples=num_samples,
        max_seq_length=args.max_seq_length,
        pad_to_max_length=False,
        shuffle_calibration_samples=False,
        trust_remote_code_model=True,
        save_compressed=True,
        output_dir=None if args.no_save else str(output_dir),
        log_dir=None if args.no_save else str(output_dir / "logs"),
    )

    if args.no_save:
        print("smoke run completed without saving")
        return

    metadata = {
        "source_model": str(model_path),
        "calibration_jsonl": str(calib_path),
        "num_calibration_samples": num_samples,
        "max_seq_length": args.max_seq_length,
        "quantization": "MixedGPTQ-W4W8A16",
        "tool": "llmcompressor",
        "model_class": class_name,
        "scheme": scheme,
        "ignore": ignore,
        "plan": plan,
        "block_size": args.block_size,
        "dampening_frac": args.dampening_frac,
        "actorder": args.actorder,
        "offload_hessians": args.offload_hessians,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    with (output_dir / "carla_quantization_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("========================================")
    print(f"saved: {output_dir}")
    print("========================================")


if __name__ == "__main__":
    main()
