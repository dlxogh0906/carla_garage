#!/usr/bin/env python3
"""Convert TF++/Qwen vertical visualization PNGs into wide camera+BEV frames.

Default output for:
  .../qwen_dev10_1/viz/RouteScenario_17569_rep0

is:
  .../qwen_dev10_1/viz_wide_camera_big/RouteScenario_17569_rep0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


CAMERA_SIZE = (2048, 768)
BEV_SIZE = (768, 768)
OUTPUT_SIZE = (CAMERA_SIZE[0] + BEV_SIZE[0], CAMERA_SIZE[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make 2816x768 wide visualization frames from vertical TF++ debug PNGs."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A RouteScenario_* folder, or a root folder containing RouteScenario_* folders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder. Defaults to sibling viz_wide_camera_big path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PNGs.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="Source camera crop height. Defaults to image_height - image_width.",
    )
    return parser.parse_args()


def scenario_dirs(input_path: Path) -> list[Path]:
    pngs = sorted(input_path.glob("*.png"))
    if pngs:
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.is_dir() and sorted(p.glob("*.png")))


def default_output_path(input_path: Path, scenarios: list[Path]) -> Path:
    if len(scenarios) == 1 and scenarios[0] == input_path:
        if input_path.parent.name == "viz":
            return input_path.parent.parent / "viz_wide_camera_big" / input_path.name
        return input_path.parent / f"{input_path.name}_wide"
    if input_path.name == "viz":
        return input_path.parent / "viz_wide_camera_big"
    return input_path.parent / f"{input_path.name}_wide"


def convert_image(src: Path, dst: Path, overwrite: bool, camera_height: int | None) -> None:
    if dst.exists() and not overwrite:
        return

    image = Image.open(src).convert("RGB")
    width, height = image.size

    if image.size == OUTPUT_SIZE:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return

    camera_h = camera_height
    if camera_h is None:
        camera_h = height - width
    camera_h = max(1, min(camera_h, height - 1))

    bev_y0 = camera_h
    bev_h = min(width, height - bev_y0)
    if bev_h <= 0:
        raise ValueError(f"Cannot find BEV crop in {src} with size {image.size}")

    camera = image.crop((0, 0, width, camera_h)).resize(CAMERA_SIZE, Image.Resampling.LANCZOS)
    bev = image.crop((0, bev_y0, width, bev_y0 + bev_h)).resize(BEV_SIZE, Image.Resampling.LANCZOS)

    out = Image.new("RGB", OUTPUT_SIZE, (255, 255, 255))
    out.paste(camera, (0, 0))
    out.paste(bev, (CAMERA_SIZE[0], 0))
    out.save(dst, quality=95)


def convert_scenario(src_dir: Path, dst_dir: Path, overwrite: bool, camera_height: int | None) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    for idx, src in enumerate(files, 1):
        convert_image(src, dst_dir / src.name, overwrite=overwrite, camera_height=camera_height)
        if idx % 100 == 0:
            print(f"{src_dir.name}: processed {idx}/{len(files)}")
    return len(files)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    scenarios = scenario_dirs(input_path)
    if not scenarios:
        raise SystemExit(f"No PNG frames found under: {input_path}")

    output_root = args.output.expanduser().resolve() if args.output else default_output_path(input_path, scenarios)
    total = 0
    for scenario in scenarios:
        if len(scenarios) == 1 and scenario == input_path:
            dst = output_root
        else:
            dst = output_root / scenario.name
        count = convert_scenario(
            scenario,
            dst,
            overwrite=args.overwrite,
            camera_height=args.camera_height,
        )
        total += count
        print(f"done {count} frames -> {dst}")

    print(f"finished {len(scenarios)} scenario(s), {total} frame(s)")


if __name__ == "__main__":
    main()
