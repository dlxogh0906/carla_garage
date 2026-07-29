#!/usr/bin/env python3
"""Render a dashboard smoke image for BEV arrow tuning."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render BEV arrow tuning smoke image.")
    parser.add_argument(
        "--dashboard-dir",
        type=Path,
        default=Path("/mnt/2/carla_garage/team_code"),
        help="Directory containing meta_action_rear_dashboard.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/2/carla_metric_result2/dashboard_renderer_smoke_arrow_tune.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--arrow-trim-px",
        type=float,
        default=32.0,
        help="Pixels trimmed from the end of the blue path body before drawing the arrow head.",
    )
    parser.add_argument(
        "--ego-clearance-m",
        type=float,
        default=1.2,
        help="Meters in front of ego vehicle before BEV path drawing starts.",
    )
    parser.add_argument(
        "--path-end-m",
        type=float,
        default=18.0,
        help="Forward distance of the final synthetic checkpoint.",
    )
    parser.add_argument(
        "--path-points",
        type=int,
        default=18,
        help="Number of synthetic checkpoints in the straight test path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.path_points < 2:
        raise ValueError("--path-points must be at least 2")

    os.environ["META_DASHBOARD_BEV_ARROW_BASE_TRIM_PX"] = str(args.arrow_trim_px)
    os.environ["META_DASHBOARD_BEV_PATH_EGO_CLEARANCE_M"] = str(args.ego_clearance_m)

    sys.path.insert(0, str(args.dashboard_dir))
    from meta_action_rear_dashboard import render_meta_action_rear_dashboard

    image = np.full((720, 1280, 3), 120, dtype=np.uint8)
    checkpoints = [(x, 0.0) for x in np.linspace(0.0, args.path_end_m, args.path_points)]
    frame = {
        "image": image,
        "rear_image": image,
        "step": 1,
        "ego_speed": 9.2,
        "ttc": 1.3,
        "ttc_threshold": 3.0,
        "intervention": True,
        "action_name": "slow_down",
        "action_idx": 1,
        "action_reason": "Traffic ahead requires slowing down.",
        "dashboard_vlm_reason": "Traffic ahead requires slowing down.",
        "multiplier": 0.6,
        "tfpp_speed_mps": 9.4,
        "final_speed_mps": 5.6,
        "pred_checkpoints": checkpoints,
        "pred_boxes": [],
        "history_key": f"arrow_tune_{args.arrow_trim_px}_{args.ego_clearance_m}",
        "reset_history": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_meta_action_rear_dashboard(frame, save_path=str(args.output))
    print(args.output)
    print(f"arrow_trim_px={args.arrow_trim_px}")
    print(f"ego_clearance_m={args.ego_clearance_m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
