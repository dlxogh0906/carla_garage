#!/usr/bin/env python3
"""Create AWQ/GPTQ W4A16 Qwen3-VL checkpoints with llm-compressor.

The calibration JSONL is produced by QwenVLMClient with QWEN_SAVE_CALIB=1.
Each row must include image_path and chat_template_text.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset, Image
from transformers import AutoConfig, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier


DEFAULT_MODEL = "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct"
DEFAULT_CALIB = (
    "/mnt/2/carla_metric_result/qwen_w8a8_calib/"
    "Qwen3-VL-8B_BF16_idx0-9/calibration_merged.jsonl"
)
DEFAULT_OUT = {
    "awq": "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-AWQ-W4A16",
    "gptq": "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct-GPTQ-W4A16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["awq", "gptq"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--calib-jsonl", default=DEFAULT_CALIB)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default depends on --method and writes under /mnt/2/pretrained_models.",
    )
    parser.add_argument("--num-samples", type=int, default=458)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-memory",
        default="",
        help="Optional accelerate max_memory, e.g. '0=36GiB,1=36GiB,cpu=240GiB'.",
    )
    parser.add_argument(
        "--offload-folder",
        default="",
        help="Optional folder for accelerate CPU/disk offload when using max_memory.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument(
        "--scheme",
        default="",
        help="Override quantization scheme. Defaults: awq=W4A16_ASYM, gptq=W4A16.",
    )
    parser.add_argument(
        "--quantize-visual",
        action="store_true",
        help="Also quantize model.visual Linear layers. Default keeps visual tower in BF16.",
    )
    parser.add_argument("--no-save", action="store_true", help="Run quantization without saving the model.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--block-size", type=int, default=128, help="GPTQ block size.")
    parser.add_argument("--dampening-frac", type=float, default=0.01, help="GPTQ dampening fraction.")
    parser.add_argument("--actorder", default="static", choices=["static", "group", "weight", "none"])
    parser.add_argument("--offload-hessians", action="store_true", help="Reduce GPTQ VRAM at the cost of speed.")
    parser.add_argument("--awq-duo-scaling", default="true", choices=["true", "false", "both"])
    parser.add_argument("--awq-n-grid", type=int, default=20)
    return parser.parse_args()


def load_calibration_dataset(path: Path, limit: int) -> Dataset:
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
                    "images": str(image_path),
                    "sample_id": item.get("sample_id", ""),
                    "prompt_mode": item.get("prompt_mode", ""),
                }
            )
            if len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No calibration rows found in {path}")
    return Dataset.from_list(rows).cast_column("images", Image(decode=True))


def qwen_vl_batch1_collator(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate one Qwen3-VL sample without adding a fake image batch dimension."""
    if len(features) != 1:
        raise ValueError("This collator is intentionally batch_size=1 only.")
    feature = features[0]
    batch: dict[str, torch.Tensor] = {}
    for key, value in feature.items():
        if key in {"sample_id", "prompt_mode"}:
            continue
        tensor = torch.as_tensor(value)
        if key in {"input_ids", "attention_mask"}:
            tensor = tensor.unsqueeze(0)
        batch[key] = tensor
    return batch


def dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def parse_max_memory(raw: str) -> dict[str, str] | None:
    if not raw:
        return None
    result: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid max-memory item: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        result[int(key) if key.isdigit() else key] = value
    return result


def pick_model_class(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", ""))
    class_names = {
        "qwen3_vl": ["Qwen3VLForConditionalGeneration"],
        "qwen3_vl_moe": ["Qwen3VLMoeForConditionalGeneration"],
        "qwen2_5_vl": ["Qwen2_5_VLForConditionalGeneration"],
        "qwen2_vl": ["Qwen2VLForConditionalGeneration"],
    }.get(model_type, [])
    class_names += ["AutoModelForImageTextToText", "AutoModelForMultimodalLM"]

    for class_name in class_names:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            return cfg, model_cls, class_name
    raise RuntimeError(f"No supported model class found for model_type={model_type}")


def default_scheme(method: str) -> str:
    # AWQ commonly uses asymmetric 4-bit weights, while GPTQ defaults to symmetric.
    return "W4A16_ASYM" if method == "awq" else "W4A16"


def parse_awq_duo_scaling(raw: str) -> bool | str:
    if raw == "both":
        return "both"
    return raw == "true"


def build_modifier(args: argparse.Namespace, ignore: list[str]):
    scheme = args.scheme or default_scheme(args.method)
    if args.method == "awq":
        return AWQModifier(
            targets="Linear",
            scheme=scheme,
            ignore=ignore,
            duo_scaling=parse_awq_duo_scaling(args.awq_duo_scaling),
            n_grid=args.awq_n_grid,
        )

    actorder = None if args.actorder == "none" else args.actorder
    return GPTQModifier(
        targets="Linear",
        scheme=scheme,
        ignore=ignore,
        block_size=args.block_size,
        dampening_frac=args.dampening_frac,
        actorder=actorder,
        offload_hessians=args.offload_hessians,
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    calib_path = Path(args.calib_jsonl)
    output_dir = Path(args.output_dir or DEFAULT_OUT[args.method])
    scheme = args.scheme or default_scheme(args.method)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.no_save:
        raise RuntimeError(f"Output dir already exists and is not empty: {output_dir}")
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_calibration_dataset(calib_path, args.num_samples)
    num_samples = min(args.num_samples, len(dataset))

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Qwen3-VL {args.method.upper()} W4A16 quantization")
    print(f"model       : {model_path}")
    print(f"calib       : {calib_path}")
    print(f"samples     : {num_samples}")
    print(f"output      : {output_dir}")
    print(f"device_map  : {args.device_map}")
    print(f"max_memory  : {args.max_memory or 'default'}")
    print(f"offload     : {args.offload_folder or 'default'}")
    print(f"dtype       : {args.dtype}")
    print(f"scheme      : {scheme}")
    print(f"visual int4 : {args.quantize_visual}")
    print(f"save        : {not args.no_save}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )
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

    model = model_cls.from_pretrained(
        str(model_path),
        **load_kwargs,
    )
    model.eval()

    ignore = ["lm_head"]
    if not args.quantize_visual:
        ignore += ["re:.*visual.*"]

    recipe = [build_modifier(args, ignore)]

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
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("smoke run completed without saving")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return

    metadata = {
        "source_model": str(model_path),
        "calibration_jsonl": str(calib_path),
        "num_calibration_samples": num_samples,
        "max_seq_length": args.max_seq_length,
        "quantization": f"{args.method.upper()}-{scheme}",
        "tool": "llmcompressor",
        "model_class": class_name,
        "quantize_visual": bool(args.quantize_visual),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if args.method == "gptq":
        metadata.update(
            {
                "block_size": args.block_size,
                "dampening_frac": args.dampening_frac,
                "actorder": args.actorder,
                "offload_hessians": args.offload_hessians,
            }
        )
    else:
        metadata.update(
            {
                "duo_scaling": args.awq_duo_scaling,
                "n_grid": args.awq_n_grid,
            }
        )
    with (output_dir / "carla_quantization_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"saved: {output_dir}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
