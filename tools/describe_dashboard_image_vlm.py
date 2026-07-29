#!/usr/bin/env python3
"""Ask Qwen3-VL to describe a saved dashboard image or crop."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _crop_dashboard(image: Image.Image, crop: str) -> Image.Image:
    width, height = image.size
    if crop == "full":
        return image
    if crop == "front":
        return image.crop((
            int(width * 0.018),
            int(height * 0.024),
            int(width * 0.496),
            int(height * 0.530),
        ))
    if crop == "bev":
        return image.crop((
            int(width * 0.506),
            int(height * 0.024),
            int(width * 0.988),
            int(height * 0.530),
        ))
    raise ValueError(f"unknown crop: {crop}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, nargs="+", help="Dashboard PNG path(s).")
    parser.add_argument("--model", required=True, help="Qwen3-VL model path.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--crop", default="front", choices=["front", "bev", "full"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--save-crop", default="")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    prompt = args.prompt.strip() or (
        "Describe the driving road situation visible in this image from the ego vehicle perspective. "
        "Focus on concrete road cues: construction obstacles, blocked lanes, vehicles, pedestrians, "
        "junctions, lane geometry, weather, and visibility. Write one factual sentence, no advice."
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()

    for image_path in args.image:
        image = Image.open(image_path).convert("RGB")
        image = _crop_dashboard(image, args.crop)
        if args.save_crop and len(args.image) == 1:
            Path(args.save_crop).parent.mkdir(parents=True, exist_ok=True)
            image.save(args.save_crop)

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(args.device)
        for key, value in list(inputs.items()):
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(dtype=dtype)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        n_in = int(inputs["input_ids"].shape[1])
        response = processor.decode(output[0][n_in:], skip_special_tokens=True).strip()
        print(f"{Path(image_path).name} [{args.crop}]: {response}")


if __name__ == "__main__":
    main()
