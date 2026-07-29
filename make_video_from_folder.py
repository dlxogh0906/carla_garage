#!/usr/bin/env python3
"""Create MP4 videos from CARLA image frame folders.

Examples
--------
python make_video_from_folder.py /path/to/RouteScenario_25381_rep0
python make_video_from_folder.py /path/to/RouteScenario_25381_rep0/dashboard --fps 10
python make_video_from_folder.py /path/to/frames -o /tmp/out.mp4 --pattern "*.jpg"
python make_video_from_folder.py /path/to/viz -o /path/to/video --overwrite
python make_video_from_folder.py /path/to/viz --subdir dashboard -o /path/to/dashboard_video
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional


class VideoJob(NamedTuple):
    name_folder: Path
    frame_folder: Path
    frames: List[Path]
    output: Path


def natural_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_frames(folder: Path, pattern: str, recursive: bool) -> List[Path]:
    globber: Iterable[Path]
    if recursive:
        globber = folder.rglob(pattern)
    else:
        globber = folder.glob(pattern)
    return sorted((p for p in globber if p.is_file()), key=natural_key)


def default_output_path(folder: Path) -> Path:
    return folder.with_suffix(".mp4")


def output_from_arg(frame_folder: Path, output_arg: Optional[Path]) -> Path:
    if output_arg is None:
        return default_output_path(frame_folder)

    output = output_arg.expanduser().resolve()
    if output.suffix.lower() == ".mp4":
        return output
    return output / f"{frame_folder.name}.mp4"


def subdir_label(subdir: Optional[str]) -> str:
    if not subdir:
        return ""
    return Path(subdir).name


def single_default_output_path(folder: Path, subdir: Optional[str]) -> Path:
    label = subdir_label(subdir)
    if label:
        return folder.parent / f"{folder.name}_{label}.mp4"
    return default_output_path(folder)


def discover_child_frame_folders(
    folder: Path,
    pattern: str,
    recursive: bool,
    subdir: Optional[str],
) -> List[VideoJob]:
    jobs: List[VideoJob] = []
    for child in sorted((p for p in folder.iterdir() if p.is_dir()), key=natural_key):
        frame_folder = child / subdir if subdir else child
        if not frame_folder.is_dir():
            continue
        frames = collect_frames(frame_folder, pattern, recursive)
        if frames:
            jobs.append(VideoJob(child, frame_folder, frames, Path()))
    return jobs


def build_jobs(args: argparse.Namespace, folder: Path) -> List[VideoJob]:
    direct_frame_folder = folder / args.subdir if args.subdir else folder
    direct_frames = (
        collect_frames(direct_frame_folder, args.pattern, args.recursive)
        if direct_frame_folder.is_dir()
        else []
    )
    child_jobs = discover_child_frame_folders(
        folder,
        args.pattern,
        args.recursive,
        args.subdir,
    )

    # If the input folder itself has frames, keep the original single-video
    # behavior. If not, treat it as a parent like ".../viz" and batch each
    # immediate child folder that contains frames.
    if direct_frames:
        output = (
            output_from_arg(folder, args.output)
            if args.output
            else single_default_output_path(folder, args.subdir)
        )
        return [VideoJob(folder, direct_frame_folder, direct_frames, output)]

    if not child_jobs:
        return []

    if args.output is None:
        output_dir = folder.parent / ("dashboard_video" if args.subdir else "video")
    else:
        output_dir = args.output.expanduser().resolve()
        if output_dir.suffix.lower() == ".mp4":
            raise ValueError("batch mode needs an output directory, not an .mp4 file")

    return [
        VideoJob(
            job.name_folder,
            job.frame_folder,
            job.frames,
            output_dir / f"{job.name_folder.name}.mp4",
        )
        for job in child_jobs
    ]


def quote_ffconcat_path(path: Path) -> str:
    # ffconcat supports single quotes and backslash escaping inside paths.
    text = str(path.resolve())
    text = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def make_video_ffmpeg(
    frames: List[Path],
    output: Path,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise FileNotFoundError("ffmpeg")

    duration = 1.0 / fps
    fd, list_name = tempfile.mkstemp(prefix="frames_", suffix=".ffconcat")
    list_path = Path(list_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for frame in frames:
                f.write(f"file {quote_ffconcat_path(frame)}\n")
                f.write(f"duration {duration:.10f}\n")
            # Repeat the final frame so the concat demuxer keeps its duration.
            f.write(f"file {quote_ffconcat_path(frames[-1])}\n")

        cmd = [
            ffmpeg_bin,
            "-y" if overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "info",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-vf",
            f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(output),
        ]
        env = os.environ.copy()
        # CARLA/conda shells often prepend conda libraries. System ffmpeg can
        # then load incompatible libpango/libglib/libncurses versions, so keep
        # ffmpeg on the normal system library path unless the caller overrides.
        env.pop("LD_LIBRARY_PATH", None)
        subprocess.run(cmd, check=True, env=env)
    finally:
        try:
            list_path.unlink()
        except FileNotFoundError:
            pass


def make_video_cv2(frames: List[Path], output: Path, fps: float, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("ffmpeg was not found, and OpenCV is not installed") from exc

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read first frame: {frames[0]}")

    height, width = first.shape[:2]
    width -= width % 2
    height -= height % 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")

    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"failed to read frame: {frame_path}")
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            elif frame.shape[1] % 2 or frame.shape[0] % 2:
                frame = frame[:height, :width]
            writer.write(frame)
    finally:
        writer.release()


def encode_video(
    frames: List[Path],
    output: Path,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg"):
        try:
            make_video_ffmpeg(frames, output, fps, crf, preset, overwrite)
            return
        except subprocess.CalledProcessError as exc:
            print(
                f"warning: ffmpeg failed with exit code {exc.returncode}; "
                "falling back to OpenCV",
                file=sys.stderr,
            )
            if output.exists() and output.stat().st_size == 0:
                output.unlink()

    make_video_cv2(frames, output, fps, overwrite)


def print_job(
    job: VideoJob,
    fps: float,
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    prefix = ""
    if index is not None and total is not None:
        prefix = f"[{index}/{total}] "
    print(f"{prefix}input folder : {job.name_folder}")
    if job.frame_folder != job.name_folder:
        print(f"{prefix}frame folder : {job.frame_folder}")
    print(f"{prefix}frames       : {len(job.frames)}")
    print(f"{prefix}first frame  : {job.frames[0].name}")
    print(f"{prefix}last frame   : {job.frames[-1].name}")
    print(f"{prefix}fps          : {fps:g}")
    print(f"{prefix}output       : {job.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MP4 video from sorted image frames in a folder.",
    )
    parser.add_argument("folder", type=Path, help="Folder containing image frames")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Single folder: output MP4 path or output directory. "
            "Parent folder: output directory. Default parent output is ../video."
        ),
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output FPS")
    parser.add_argument("--pattern", default="*.png", help='Frame glob pattern, e.g. "*.png"')
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search frames recursively. Off by default so dashboard/ is not mixed in.",
    )
    parser.add_argument(
        "--subdir",
        help=(
            "Use this subdirectory inside each scenario folder as the frame source, "
            'e.g. "dashboard". Useful for batch dashboard videos from a viz folder.'
        ),
    )
    parser.add_argument("--crf", type=int, default=18, help="ffmpeg x264 quality, lower is better")
    parser.add_argument("--preset", default="medium", help="ffmpeg x264 preset")
    parser.add_argument("--overwrite", action="store_true", help="Replace output if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be used")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"error: folder does not exist: {folder}", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("error: --fps must be positive", file=sys.stderr)
        return 2

    try:
        jobs = build_jobs(args, folder)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not jobs:
        print(
            f"error: no frames matched {args.pattern!r} in {folder} "
            "or its immediate child folders",
            file=sys.stderr,
        )
        return 2

    print(f"pattern      : {args.pattern}")
    print(f"jobs         : {len(jobs)}")

    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        print_job(job, args.fps, idx if total > 1 else None, total if total > 1 else None)
        if args.dry_run:
            continue
        if job.output.exists() and not args.overwrite:
            print(f"skip existing: {job.output}  (pass --overwrite to replace)")
            continue
        encode_video(job.frames, job.output, args.fps, args.crf, args.preset, args.overwrite)
        print(f"done: {job.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
