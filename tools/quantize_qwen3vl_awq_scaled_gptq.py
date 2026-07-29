#!/usr/bin/env python3
"""Create an AWQ-scaled GPTQ W4A16 Qwen3-VL checkpoint.

This is a GPTQ-based quantizer with AWQ-style preprocessing:

1. Run AWQ calibration only to find channel-wise smoothing scales via grid search.
2. Fold the selected scales into the floating-point model weights.
3. Remove AWQ quantization metadata so the final quantizer is not AWQ.
4. Run GPTQ W4A16 on the scaled model and save a compressed-tensors checkpoint.

No W8 layer protection is used. The final quantized language Linear weights are W4A16
GPTQ. By default, visual modules and lm_head are preserved just like the existing
W4A16 scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoProcessor

from quantize_qwen3vl_w4a16 import (
    DEFAULT_CALIB,
    DEFAULT_MODEL,
    dtype_from_name,
    load_calibration_dataset,
    parse_max_memory,
    pick_model_class,
    qwen_vl_batch1_collator,
)


DEFAULT_OUT = "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQScaledGPTQ-W4A16-n64"


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
    parser.add_argument("--no-save", action="store_true", help="Run both phases without saving.")
    parser.add_argument("--dry-plan", action="store_true", help="Print plan and exit.")
    parser.add_argument(
        "--quantize-visual",
        action="store_true",
        help="Also quantize visual Linear layers to GPTQ W4A16. Default keeps visual tower BF16.",
    )

    parser.add_argument(
        "--awq-search-scheme",
        default="W4A16",
        help=(
            "Temporary quantization scheme used only inside AWQ scale search. "
            "Default matches final GPTQ W4A16. W4A16_ASYM is also useful for AWQ-like search."
        ),
    )
    parser.add_argument("--awq-duo-scaling", default="true", choices=["true", "false", "both"])
    parser.add_argument("--awq-n-grid", type=int, default=20)

    parser.add_argument("--gptq-scheme", default="W4A16")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--actorder", default="static", choices=["static", "group", "weight", "none"])
    parser.add_argument("--offload-hessians", action="store_true")
    return parser.parse_args()


def parse_awq_duo_scaling(raw: str) -> bool | str:
    if raw == "both":
        return "both"
    return raw == "true"


def build_ignore(quantize_visual: bool) -> list[str]:
    ignore = ["lm_head"]
    if not quantize_visual:
        ignore.append("re:.*visual.*")
    return ignore


def clear_quantization_artifacts(model) -> dict[str, int]:
    """Remove temporary AWQ quantization state while keeping smoothed weights.

    AWQModifier folds the selected smoothing scales into floating-point weights.
    The final quantizer should be GPTQ, so stale AWQ quantization schemes, observers,
    and qparams must not leak into the GPTQ phase.
    """

    attr_names = [
        "quantization_scheme",
        "quantization_status",
        "quantization_enabled",
        "weight_scale",
        "weight_zero_point",
        "weight_g_idx",
        "weight_global_scale",
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
        "q_scale",
        "q_zero_point",
        "k_scale",
        "k_zero_point",
        "v_scale",
        "v_zero_point",
    ]
    observer_names = [
        "weight_observer",
        "input_observer",
        "output_observer",
        "q_observer",
        "k_observer",
        "v_observer",
    ]

    removed: dict[str, int] = {name: 0 for name in attr_names + observer_names}
    for module in model.modules():
        for name in attr_names + observer_names:
            if hasattr(module, name):
                delattr(module, name)
                removed[name] += 1
    return {name: count for name, count in removed.items() if count}


def print_plan(args: argparse.Namespace, ignore: list[str]) -> None:
    print("========================================")
    print("Qwen3-VL AWQ-Scaled GPTQ W4A16 plan")
    print(f"model              : {args.model}")
    print(f"calib              : {args.calib_jsonl}")
    print(f"samples            : {args.num_samples}")
    print(f"output             : {args.output_dir}")
    print(f"visual             : {'GPTQ W4A16' if args.quantize_visual else 'BF16 / ignored'}")
    print("phase 1            : AWQ scale search only")
    print(f"  search scheme    : {args.awq_search_scheme}")
    print(f"  duo scaling      : {args.awq_duo_scaling}")
    print(f"  n_grid           : {args.awq_n_grid}")
    print("phase 2            : GPTQ W4A16 final quantization")
    print(f"  scheme           : {args.gptq_scheme}")
    print(f"  block_size       : {args.block_size}")
    print(f"  dampening_frac   : {args.dampening_frac}")
    print(f"  actorder         : {args.actorder}")
    print(f"  offload_hessians : {args.offload_hessians}")
    print(f"ignore             : {ignore}")
    print("W8 protection      : none")
    print("========================================")


def run_awq_scaling_phase(model, processor, dataset, args: argparse.Namespace, ignore: list[str]) -> None:
    awq_modifier = AWQModifier(
        targets="Linear",
        scheme=args.awq_search_scheme,
        ignore=ignore,
        duo_scaling=parse_awq_duo_scaling(args.awq_duo_scaling),
        n_grid=args.awq_n_grid,
    )

    print("========================================")
    print("Phase 1/2: AWQ-style scale search + weight smoothing")
    print("This phase does not save an AWQ checkpoint.")
    print("========================================")
    oneshot(
        model=model,
        processor=processor,
        dataset=dataset,
        recipe=[awq_modifier],
        batch_size=1,
        data_collator=qwen_vl_batch1_collator,
        num_calibration_samples=min(args.num_samples, len(dataset)),
        max_seq_length=args.max_seq_length,
        pad_to_max_length=False,
        shuffle_calibration_samples=False,
        trust_remote_code_model=True,
        save_compressed=False,
        clear_sparse_session=True,
        output_dir=None,
        log_dir=None,
    )


def run_gptq_phase(
    model,
    processor,
    dataset,
    args: argparse.Namespace,
    ignore: list[str],
    output_dir: Path,
) -> None:
    actorder = None if args.actorder == "none" else args.actorder
    gptq_modifier = GPTQModifier(
        targets="Linear",
        scheme=args.gptq_scheme,
        ignore=ignore,
        block_size=args.block_size,
        dampening_frac=args.dampening_frac,
        actorder=actorder,
        offload_hessians=args.offload_hessians,
    )

    print("========================================")
    print("Phase 2/2: GPTQ W4A16 final quantization")
    print("========================================")
    oneshot(
        model=model,
        processor=processor,
        dataset=dataset,
        recipe=[gptq_modifier],
        batch_size=1,
        data_collator=qwen_vl_batch1_collator,
        num_calibration_samples=min(args.num_samples, len(dataset)),
        max_seq_length=args.max_seq_length,
        pad_to_max_length=False,
        shuffle_calibration_samples=False,
        trust_remote_code_model=True,
        save_compressed=True,
        clear_sparse_session=True,
        output_dir=None if args.no_save else str(output_dir),
        log_dir=None if args.no_save else str(output_dir / "logs"),
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    calib_path = Path(args.calib_jsonl)
    output_dir = Path(args.output_dir)
    ignore = build_ignore(args.quantize_visual)

    print_plan(args, ignore)
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
    load_kwargs: dict[str, Any] = {
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

    run_awq_scaling_phase(model, processor, dataset, args, ignore)
    removed = clear_quantization_artifacts(model)
    print("========================================")
    print("Cleared temporary AWQ quantization artifacts")
    print(json.dumps(removed, indent=2, sort_keys=True))
    print("========================================")

    run_gptq_phase(model, processor, dataset, args, ignore, output_dir)

    if args.no_save:
        print("========================================")
        print("smoke run completed without saving")
        print("========================================")
        return

    metadata = {
        "source_model": str(model_path),
        "calibration_jsonl": str(calib_path),
        "num_calibration_samples": num_samples,
        "max_seq_length": args.max_seq_length,
        "quantization": "AWQScaledGPTQ-W4A16",
        "tool": "llmcompressor",
        "model_class": class_name,
        "algorithm": {
            "phase_1": "AWQ-style channel-wise scale search and weight smoothing only",
            "phase_2": "GPTQ W4A16 final quantization",
            "w8_protection": False,
            "final_quantizer": "GPTQ",
        },
        "awq_scaling": {
            "scheme": args.awq_search_scheme,
            "duo_scaling": args.awq_duo_scaling,
            "n_grid": args.awq_n_grid,
        },
        "gptq": {
            "scheme": args.gptq_scheme,
            "block_size": args.block_size,
            "dampening_frac": args.dampening_frac,
            "actorder": args.actorder,
            "offload_hessians": args.offload_hessians,
        },
        "quantize_visual": bool(args.quantize_visual),
        "ignore": ignore,
        "cleared_awq_artifacts": removed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    with (output_dir / "carla_quantization_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("========================================")
    print(f"saved: {output_dir}")
    print("========================================")


if __name__ == "__main__":
    main()
