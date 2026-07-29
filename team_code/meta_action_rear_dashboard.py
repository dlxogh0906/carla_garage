"""
Presentation dashboard for TF++ + TTC-gated Meta-Action VLA.

The renderer intentionally uses only data already available in the online
agent loop: front/rear RGB, LiDAR BEV, predicted checkpoints, detected boxes,
TTC, speed, and the cached VLA meta-action. It avoids presentation-only fields
such as weather or wall-clock time.
"""
from __future__ import annotations

import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


BG = "#030b12"
PANEL = "#07131d"
EDGE = "#2b4053"
EDGE_SOFT = "#1d3142"
TEXT = "#f4f8fc"
MUTED = "#9caebf"
BLUE = "#2aa8ff"
CYAN = "#31e4ff"
VIOLET = "#7a6cff"
GREEN = "#7be34f"
ORANGE = "#ff9f1c"
RED = "#ff4d3d"
PATH_GRADIENT = (CYAN, BLUE, VIOLET)

_META_COLORS = {
    "proceed": GREEN,
    "slow_down": ORANGE,
    "stop": RED,
    "yield": "#ffb84d",
    "turn_left": BLUE,
    "turn_right": BLUE,
    "change_lane_left": "#b779ff",
    "change_lane_right": "#b779ff",
}

_ACTION_LABELS = {
    "proceed": "Keep Plan",
    "slow_down": "Slow Down",
    "stop": "Stop",
    "yield": "Yield",
    "turn_left": "Turn Left",
    "turn_right": "Turn Right",
    "change_lane_left": "Lane Left",
    "change_lane_right": "Lane Right",
}

_CLASS_CUE_LABELS = {
    0: "vehicle",
    1: "pedestrian",
    2: "red traffic light",
    3: "stop sign",
    4: "emergency vehicle",
}

_CLASS_COLORS = {
    0: (118, 139, 157),  # vehicle
    1: (255, 82, 65),    # pedestrian
    2: (255, 50, 50),    # red light
    3: (255, 174, 88),   # stop sign
    4: (35, 207, 207),   # emergency vehicle
}

# Match the CARLA ego footprint used by the controller.
EGO_HALF_LENGTH_M = 2.4508416652679443
EGO_HALF_WIDTH_M = 1.0641621351242065
BEV_PATH_EGO_CLEARANCE_M = 1.2
BEV_ARROW_BASE_TRIM_PX = 32.0

_HISTORY: Dict[str, Dict[str, Any]] = {}


def _get_history(key: str, reset: bool = False) -> Dict[str, Any]:
    if reset or key not in _HISTORY:
        _HISTORY[key] = {
            "ttc": [],
            "multiplier": [],
            "step": [],
            "action_idx": [],
            "ego_speed": [],
            "final_speed": [],
            "quote_held_text": "",
            "quote_last_update_step": -10**9,
        }
    return _HISTORY[key]


def render_meta_action_rear_dashboard(
    frame_data: Dict[str, Any],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Render and optionally save a presentation-ready dashboard image."""
    key = str(frame_data.get("history_key", "default"))
    hist = _get_history(key, reset=bool(frame_data.get("reset_history", False)))

    step = int(frame_data.get("step", 0))
    ttc = float(frame_data.get("ttc", 999.0))
    multiplier = float(frame_data.get("multiplier", 1.0))
    action_idx = int(frame_data.get("action_idx", 0))
    ego_speed = float(frame_data.get("ego_speed", 0.0))
    final_speed = float(frame_data.get("final_speed_mps", float("nan")))

    hist["step"].append(step)
    hist["ttc"].append(min(ttc, 10.0) if ttc < 999.0 else float("nan"))
    hist["multiplier"].append(multiplier)
    hist["action_idx"].append(action_idx)
    hist["ego_speed"].append(ego_speed)
    hist["final_speed"].append(final_speed if math.isfinite(final_speed) else float("nan"))

    fig = plt.figure(figsize=(18, 9), dpi=120, facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        4,
        height_ratios=[1.12, 0.82],
        width_ratios=[1.08, 0.92, 1.0, 1.0],
        left=0.018,
        right=0.982,
        top=0.975,
        bottom=0.025,
        hspace=0.055,
        wspace=0.045,
    )

    ax_front = fig.add_subplot(gs[0, 0:2])
    ax_bev = fig.add_subplot(gs[0, 2:4])
    ax_rear = fig.add_subplot(gs[1, 0])
    ax_state = fig.add_subplot(gs[1, 1])
    ax_explain = fig.add_subplot(gs[1, 2:4])

    _draw_camera_panel(ax_front, frame_data, "Front View", "image", primary=True)
    _draw_bev_panel(ax_bev, frame_data)
    _draw_camera_panel(ax_rear, frame_data, "Rear View", "rear_image", primary=False)
    _draw_driving_state(ax_state, frame_data, hist)
    _draw_explainability(ax_explain, frame_data, hist)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_fig, h_fig = fig.canvas.get_width_height()
    result = buf.reshape(h_fig, w_fig, 4)[:, :, :3].copy()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    plt.close(fig)
    return result


def _draw_camera_panel(
    ax: plt.Axes,
    fd: Dict[str, Any],
    title: str,
    key: str,
    primary: bool,
) -> None:
    _clear_panel(ax, facecolor=PANEL)
    _panel_title(ax, title)

    img = _image_from_frame(fd.get(key))
    if img is None:
        ax.text(0.5, 0.5, f"No {title.lower()} image", color=MUTED, ha="center", va="center", transform=ax.transAxes)
        return

    h, w = img.shape[:2]

    if primary or key == "rear_image":
        img = _remove_front_green_markers(img)

    ax.imshow(img)
    ax.set_aspect("auto")

    if primary:
        _draw_camera_path(ax, fd, w, h)
    accent = ORANGE if bool(fd.get("intervention", False)) and primary else EDGE
    for spine in ax.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(1.5 if accent == RED else 1.0)


def _draw_camera_path(ax: plt.Axes, fd: Dict[str, Any], width: int, height: int) -> None:
    checkpoints = _normalize_points(fd.get("pred_checkpoints"))
    if checkpoints is None:
        return

    points = []
    for x_m, y_m in checkpoints[:12]:
        forward = max(float(x_m), 0.0)
        lateral = float(y_m)
        depth = 1.0 + 0.16 * forward
        px = width * 0.50 + (lateral * width * 0.11) / depth
        py = height * 0.94 - (forward * height * 0.075) / depth
        points.append((px, py))

    if len(points) < 2:
        return

    _draw_projected_path_ribbon(ax, np.asarray(points, dtype=np.float32), width)


def _draw_projected_path_ribbon(ax: plt.Axes, points: np.ndarray, image_width: int) -> None:
    if len(points) < 2:
        return

    points = _densify_polyline(points, samples=56)
    if len(points) < 2:
        return

    center_segments = np.stack([points[:-1], points[1:]], axis=1)
    ax.add_collection(LineCollection(center_segments, colors=["#07151d"], linewidths=7.5, alpha=0.45,
                                     capstyle="round", zorder=5))
    ax.add_collection(LineCollection(center_segments, colors=[BLUE], linewidths=4.6, alpha=0.90,
                                     capstyle="round", zorder=6))
    ax.add_collection(LineCollection(center_segments, colors=[CYAN], linewidths=1.8, alpha=0.38,
                                     capstyle="round", zorder=7))

    dot_count = int(np.clip(round(image_width / 58.0), 7, 12))
    dot_idx = np.unique(np.linspace(0, len(points) - 1, dot_count, dtype=np.int32))
    dots = points[dot_idx]
    ax.scatter(dots[:, 0], dots[:, 1], s=58, c="#04131d", edgecolors="none", alpha=0.52, zorder=8)
    ax.scatter(dots[:, 0], dots[:, 1], s=34, c=CYAN, edgecolors=BLUE, linewidths=1.1, alpha=0.96, zorder=9)


def _draw_path_arrow_tip(ax: plt.Axes, points: np.ndarray) -> None:
    distances = np.zeros(len(points), dtype=np.float32)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    total = float(distances[-1])
    if total <= 1.0:
        return

    start, _, _ = _sample_ribbon_point(points, np.ones(len(points), dtype=np.float32), distances, total * 0.86)
    end = points[-1]
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=28,
            linewidth=8.0,
            color=BLUE,
            alpha=0.48,
            shrinkA=0,
            shrinkB=0,
            zorder=8,
        )
    )
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.5,
            color=CYAN,
            alpha=0.42,
            shrinkA=0,
            shrinkB=0,
            zorder=9,
        )
    )


def _sample_ribbon_point(
    points: np.ndarray,
    widths: np.ndarray,
    distances: np.ndarray,
    target_distance: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    idx = int(np.searchsorted(distances, target_distance, side="right") - 1)
    idx = int(np.clip(idx, 0, len(points) - 2))
    segment_len = max(float(distances[idx + 1] - distances[idx]), 1e-6)
    t = float(np.clip((target_distance - distances[idx]) / segment_len, 0.0, 1.0))
    point = points[idx] * (1.0 - t) + points[idx + 1] * t
    tangent = points[idx + 1] - points[idx]
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
    width = float(widths[idx] * (1.0 - t) + widths[idx + 1] * t)
    return point, tangent, width


def _densify_polyline(points: np.ndarray, samples: int) -> np.ndarray:
    distances = np.zeros(len(points), dtype=np.float32)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    total = float(distances[-1])
    if total <= 1.0:
        return points
    target = np.linspace(0.0, total, max(samples, len(points)))
    xs = np.interp(target, distances, points[:, 0])
    ys = np.interp(target, distances, points[:, 1])
    return np.column_stack((xs, ys)).astype(np.float32)


def _draw_gradient_arrow(ax: plt.Axes, points: np.ndarray) -> None:
    if len(points) < 2:
        return

    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = [_gradient_hex(i / max(len(segments) - 1, 1)) for i in range(len(segments))]
    glow = LineCollection(segments, colors=colors, linewidths=7.0, alpha=0.24, capstyle="round", zorder=4)
    core = LineCollection(segments, colors=colors, linewidths=3.2, alpha=0.96, capstyle="round", zorder=5)
    ax.add_collection(glow)
    ax.add_collection(core)

    if len(points) >= 3:
        start = points[-3]
        end = points[-1]
    else:
        start = points[-2]
        end = points[-1]
    arrow = mpatches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=24,
        linewidth=2.6,
        color=PATH_GRADIENT[-1],
        alpha=0.98,
        shrinkA=0,
        shrinkB=0,
        zorder=6,
    )
    ax.add_patch(arrow)


def _gradient_hex(t: float) -> str:
    if t <= 0.5:
        return _mix_hex(PATH_GRADIENT[0], PATH_GRADIENT[1], t * 2.0)
    return _mix_hex(PATH_GRADIENT[1], PATH_GRADIENT[2], (t - 0.5) * 2.0)


def _gradient_rgb(t: float) -> Tuple[int, int, int]:
    return _hex_to_rgb(_gradient_hex(t))


def _mix_hex(a: str, b: str, t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    ca = np.asarray(_hex_to_rgb(a), dtype=np.float32)
    cb = np.asarray(_hex_to_rgb(b), dtype=np.float32)
    c = np.round(ca * (1.0 - t) + cb * t).astype(np.uint8)
    return "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _draw_bev_panel(ax: plt.Axes, fd: Dict[str, Any]) -> None:
    _clear_panel(ax, facecolor=PANEL)
    _panel_title(ax, "Bird-Eye View")

    bev = _make_bev_canvas(fd)
    ax.imshow(bev)
    ax.set_aspect("auto")


def _make_bev_canvas(fd: Dict[str, Any]) -> np.ndarray:
    height, width = 620, 760
    scene = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_scene_surface(scene)

    _draw_bev_grid(scene)
    _draw_lidar_bev(scene, fd.get("lidar_bev"))
    _draw_bev_semantic_lanes(scene, fd.get("pred_bev_semantic"))
    _draw_ttc_gate_zone(scene, bool(fd.get("intervention", False)))
    _draw_bev_checkpoints(scene, fd.get("pred_checkpoints"))
    _draw_bev_boxes(scene, fd.get("pred_boxes"))
    _draw_ego(scene)
    return scene


def _draw_scene_surface(canvas: np.ndarray) -> None:
    """Set the dark LiDAR-style BEV plane used in the final BEV reference."""
    canvas[:, :, :] = (4, 14, 22)


def _project_bev_scene(scene: np.ndarray) -> np.ndarray:
    """Project the metric top view into a clean elevated-camera presentation view."""
    height, width = scene.shape[:2]
    rear_crop = float(min(height - 1, _bev_center(scene)[1] + 55))
    source = np.float32([
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), rear_crop],
        [0.0, rear_crop],
    ])
    target = np.float32([
        [width * 0.33, height * 0.07],
        [width * 0.67, height * 0.07],
        [width * 1.08, height * 1.04],
        [-width * 0.08, height * 1.04],
    ])
    matrix = cv2.getPerspectiveTransform(source, target)
    output = cv2.warpPerspective(
        scene,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(239, 239, 238),
    )

    return output


def _draw_bev_grid(canvas: np.ndarray) -> None:
    center = _bev_center(canvas)
    scale = _bev_scale(canvas)

    for meters in range(-30, 31, 10):
        x = int(round(center[0] + meters * scale))
        y = int(round(center[1] - meters * scale))
        cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (35, 52, 66), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (35, 52, 66), 1, cv2.LINE_AA)


def _draw_bev_semantic_lanes(canvas: np.ndarray, semantic: Any) -> None:
    """Render the same predicted lane classes used by TF++'s native BEV view."""
    arr = _to_numpy(semantic)
    if arr is None:
        return
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        classes = np.argmax(arr, axis=0)
    elif arr.ndim == 2:
        classes = arr
    else:
        return

    solid_mask = _thin_binary_mask(classes == 3)
    divider_mask = _thin_binary_mask(classes == 4)
    if not np.any(solid_mask) and not np.any(divider_mask):
        return

    _draw_smooth_lane_mask(
        canvas,
        solid_mask,
        edge_color=(204, 207, 209),
        core_color=(168, 172, 174),
        edge_width=4,
        core_width=2,
    )
    _draw_smooth_lane_mask(
        canvas,
        divider_mask,
        edge_color=(216, 218, 220),
        core_color=(186, 189, 191),
        edge_width=3,
        core_width=1,
    )


def _draw_smooth_lane_mask(
    canvas: np.ndarray,
    mask: np.ndarray,
    edge_color: Tuple[int, int, int],
    core_color: Tuple[int, int, int],
    edge_width: int,
    core_width: int,
) -> None:
    thinned = (_thin_binary_mask(mask) > 0).astype(np.uint8)
    warped = _warp_bev_mask(canvas, thinned, ppm=4.0)
    warped = (warped > 0).astype(np.uint8)
    if not np.any(warped):
        return

    bridge = cv2.dilate(warped, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridge, connectivity=8)
    lane_lines: List[np.ndarray] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 7:
            continue
        ys, xs = np.where(labels == label)
        line = _fit_smooth_lane_line(xs.astype(np.float32), ys.astype(np.float32))
        if line is not None:
            lane_lines.append(line)

    for line in lane_lines:
        cv2.polylines(canvas, [line], False, edge_color, edge_width, cv2.LINE_AA)
    for line in lane_lines:
        cv2.polylines(canvas, [line], False, core_color, core_width, cv2.LINE_AA)


def _fit_smooth_lane_line(xs: np.ndarray, ys: np.ndarray) -> Optional[np.ndarray]:
    if xs.size < 4:
        return None
    span_x = float(xs.max() - xs.min())
    span_y = float(ys.max() - ys.min())
    if max(span_x, span_y) < 5.0:
        return None

    major = ys if span_y >= span_x else xs
    minor = xs if span_y >= span_x else ys
    order = np.argsort(major)
    major = major[order]
    minor = minor[order]

    bin_count = int(np.clip(max(span_x, span_y) / 7.0, 4, 28))
    bins = np.linspace(float(major.min()), float(major.max()), bin_count + 1)
    smooth_major: List[float] = []
    smooth_minor: List[float] = []
    for start, end in zip(bins[:-1], bins[1:]):
        pick = (major >= start) & (major <= end)
        if np.count_nonzero(pick) == 0:
            continue
        smooth_major.append(float(np.mean(major[pick])))
        smooth_minor.append(float(np.mean(minor[pick])))

    if len(smooth_major) < 2:
        return None

    major_arr = np.asarray(smooth_major, dtype=np.float32)
    minor_arr = np.asarray(smooth_minor, dtype=np.float32)
    coeff = np.polyfit(major_arr, minor_arr, 1)
    sample = np.linspace(float(major_arr.min()), float(major_arr.max()), max(8, len(major_arr) * 3))
    fitted = np.polyval(coeff, sample)

    if span_y >= span_x:
        points = np.column_stack((fitted, sample))
    else:
        points = np.column_stack((sample, fitted))
    return np.round(points).astype(np.int32).reshape((-1, 1, 2))


def _thin_binary_mask(mask: np.ndarray) -> np.ndarray:
    work = (np.asarray(mask, dtype=np.uint8) * 255).copy()
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return skeleton


def _warp_bev_mask(canvas: np.ndarray, mask: np.ndarray, ppm: float) -> np.ndarray:
    source_h, source_w = mask.shape
    center = _bev_center(canvas)
    ratio = _bev_scale(canvas) / ppm
    matrix = np.array(
        [
            [0.0, ratio, center[0] - (source_h * 0.5 - 0.5) * ratio],
            [-ratio, 0.0, center[1] + (source_w * 0.5 - 0.5) * ratio],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(mask, matrix, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST)


def _draw_lidar_bev(canvas: np.ndarray, lidar_bev: Any) -> None:
    arr = _to_numpy(lidar_bev)
    if arr is None:
        return
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return

    values = np.asarray(arr, dtype=np.float32)
    if values.size == 0:
        return
    vmax = float(np.nanpercentile(values, 99.5))
    if vmax <= 1e-6:
        return
    values = np.clip(values / vmax, 0.0, 1.0)

    rows, cols = np.nonzero(values > 0.04)
    if len(rows) == 0:
        return
    if len(rows) > 4500:
        pick = np.linspace(0, len(rows) - 1, 4500, dtype=np.int32)
        rows = rows[pick]
        cols = cols[pick]

    min_x, min_y, ppm = -32.0, -32.0, 4.0
    x_m = min_x + (cols.astype(np.float32) + 0.5) / ppm
    y_m = min_y + (rows.astype(np.float32) + 0.5) / ppm
    weights = values[rows, cols]

    for x, y, weight in zip(x_m, y_m, weights):
        px, py = _vehicle_to_bev_px(canvas, x, y)
        if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
            c = int(115 + 130 * float(weight))
            c = int(np.clip(c, 115, 248))
            cv2.circle(canvas, (px, py), 1, (c, min(c + 5, 255), min(c + 9, 255)), -1, cv2.LINE_AA)


def _draw_ttc_gate_zone(canvas: np.ndarray, active: bool) -> None:
    if not active:
        return
    center = _bev_center(canvas)
    scale = _bev_scale(canvas)
    pts = np.array(
        [
            center,
            (int(center[0] - 6.0 * scale), int(center[1] - 22.0 * scale)),
            (int(center[0] + 6.0 * scale), int(center[1] - 22.0 * scale)),
        ],
        dtype=np.int32,
    )
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, pts, (92, 59, 22))
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, dst=canvas)
    cv2.polylines(canvas, [pts], True, (224, 164, 82), 1, cv2.LINE_AA)


def _draw_bev_checkpoints(canvas: np.ndarray, checkpoints: Any) -> None:
    points_m = _normalize_points(checkpoints)
    if points_m is None:
        return

    path_start_x_m = EGO_HALF_LENGTH_M + _env_float(
        "META_DASHBOARD_BEV_PATH_EGO_CLEARANCE_M",
        BEV_PATH_EGO_CLEARANCE_M,
    )
    points_ahead = [(float(x), float(y)) for x, y in points_m[:18] if float(x) >= path_start_x_m]
    points = [_vehicle_to_bev_px(canvas, x, y) for x, y in points_ahead]
    points = [(x, y) for x, y in points if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]]
    if len(points) < 2:
        return

    arrow_base_trim_px = _env_float("META_DASHBOARD_BEV_ARROW_BASE_TRIM_PX", BEV_ARROW_BASE_TRIM_PX)
    shaft_points = _trim_path_tail(points, arrow_base_trim_px)
    overlay = canvas.copy()
    ribbon = _flat_path_ribbon(shaft_points, width=12.0)
    if ribbon is None:
        return
    cv2.fillPoly(overlay, [ribbon], (32, 128, 231), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0.0, dst=canvas)
    _draw_bev_path_tip(canvas, points)


def _draw_bev_path_tip(canvas: np.ndarray, points: List[Tuple[int, int]]) -> None:
    if len(points) < 2:
        return
    tip = np.asarray(points[-1], dtype=np.float32)
    prev = np.asarray(points[-2], dtype=np.float32)
    direction = tip - prev
    length = float(np.linalg.norm(direction))
    if length < 1e-4:
        return
    direction /= length
    normal = np.array((-direction[1], direction[0]), dtype=np.float32)
    arrow_len = 17.0
    arrow_half = 10.0
    arrow = np.asarray(
        [
            tip + direction * 7.0,
            tip - direction * arrow_len + normal * arrow_half,
            tip - direction * arrow_len - normal * arrow_half,
        ],
        dtype=np.float32,
    )
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, np.round(arrow).astype(np.int32), (37, 149, 239), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, dst=canvas)


def _trim_path_tail(points: List[Tuple[int, int]], trim_px: float) -> List[Tuple[int, int]]:
    if len(points) < 2 or trim_px <= 0:
        return points

    path = [np.asarray(point, dtype=np.float32) for point in points]
    remaining = float(trim_px)
    for idx in range(len(path) - 1, 0, -1):
        segment = path[idx] - path[idx - 1]
        length = float(np.linalg.norm(segment))
        if length < 1e-4:
            continue
        if remaining < length:
            new_end = path[idx] - segment / length * remaining
            return [(int(round(p[0])), int(round(p[1]))) for p in path[:idx] + [new_end]]
        remaining -= length

    return points[:2]


def _flat_path_ribbon(points: List[Tuple[int, int]], width: float) -> Optional[np.ndarray]:
    path = np.asarray(points, dtype=np.float32)
    if path.shape[0] < 2:
        return None

    deduped = [path[0]]
    for point in path[1:]:
        if float(np.linalg.norm(point - deduped[-1])) >= 1.5:
            deduped.append(point)
    path = np.asarray(deduped, dtype=np.float32)
    if path.shape[0] < 2:
        return None

    half_width = width * 0.5
    normals: List[np.ndarray] = []
    for idx in range(path.shape[0]):
        if idx == 0:
            tangent = path[1] - path[0]
        elif idx == path.shape[0] - 1:
            tangent = path[-1] - path[-2]
        else:
            tangent = path[idx + 1] - path[idx - 1]
        length = float(np.linalg.norm(tangent))
        if length < 1e-4:
            normal = np.array((0.0, 1.0), dtype=np.float32)
        else:
            tangent = tangent / length
            normal = np.array((-tangent[1], tangent[0]), dtype=np.float32)
        normals.append(normal)

    normal_arr = np.asarray(normals, dtype=np.float32)
    left = path + normal_arr * half_width
    right = path - normal_arr * half_width
    ribbon = np.vstack((left, right[::-1]))
    return np.round(ribbon).astype(np.int32)


def _draw_bev_boxes(canvas: np.ndarray, boxes: Any) -> None:
    arr = _to_numpy(boxes)
    if arr is None:
        return
    arr = np.asarray(arr, dtype=object if np.asarray(arr).dtype == object else np.float32)
    if arr.size == 0:
        return
    if arr.dtype == object:
        rows = []
        for row in arr:
            row_arr = _to_numpy(row)
            if row_arr is not None:
                rows.append(np.asarray(row_arr, dtype=np.float32).reshape(-1))
        if not rows:
            return
        arr = np.asarray(rows, dtype=np.float32)
    else:
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[1] < 8:
        return

    debug = os.environ.get("META_DASHBOARD_DEBUG_BBOX", "0").strip() == "1"
    centers = np.asarray(arr[:, :2], dtype=np.float32)
    extents = np.asarray(arr[:, 2:4], dtype=np.float32)
    finite = np.isfinite(centers).all(axis=1) & np.isfinite(extents).all(axis=1)
    image_space = False
    if np.any(finite):
        max_center = float(np.nanmax(np.abs(centers[finite])))
        max_extent = float(np.nanmax(np.abs(extents[finite])))
        image_space = max_center > 80.0 or max_extent > 12.0
    if image_space:
        drawn, skipped = _draw_bev_boxes_image_space(canvas, arr)
        if debug:
            sample = arr[:3, :8].tolist()
            print(
                f"[MetaDash] bbox render image-space rows={len(arr)} drawn={drawn} "
                f"skipped={skipped} sample={sample}",
                flush=True,
            )
        return

    drawn = 0
    skipped = 0
    for idx, box in enumerate(arr[:48]):
        x_m = float(box[0])
        y_m = float(box[1])
        if not (np.isfinite(x_m) and np.isfinite(y_m)):
            skipped += 1
            continue
        class_id = int(box[7])
        half_length = float(box[2])
        half_width = float(box[3])
        if not (np.isfinite(half_length) and np.isfinite(half_width)) or half_length <= 0.0 or half_width <= 0.0:
            skipped += 1
            continue
        if abs(x_m) > 45.0 or abs(y_m) > 45.0:
            skipped += 1
            continue
        yaw = float(box[4]) if len(box) > 4 else 0.0
        color = (116, 151, 171)
        fill = None
        _draw_bev_box_outline(canvas, x_m, y_m, half_length, half_width, yaw, color, fill, thickness=2)
        drawn += 1
        if class_id == 1:
            _draw_scene_pedestrian(canvas, x_m, y_m)
    if debug:
        sample = arr[:3, :8].tolist()
        print(f"[MetaDash] bbox render rows={len(arr)} drawn={drawn} skipped={skipped} sample={sample}", flush=True)


def _draw_bev_boxes_image_space(canvas: np.ndarray, arr: np.ndarray) -> Tuple[int, int]:
    native_size = float(_env_float("META_DASHBOARD_BBOX_IMAGE_SIZE", 1024.0))
    if native_size <= 1.0:
        native_size = 1024.0
    sx = canvas.shape[1] / native_size
    sy = canvas.shape[0] / native_size
    drawn = 0
    skipped = 0
    for box in arr[:48]:
        row = float(box[0])
        col = float(box[1])
        half_row = float(box[2])
        half_col = float(box[3])
        if not all(np.isfinite(v) for v in (row, col, half_row, half_col)):
            skipped += 1
            continue
        if half_row <= 0.0 or half_col <= 0.0:
            skipped += 1
            continue
        if row < -half_row or col < -half_col or row > native_size + half_row or col > native_size + half_col:
            skipped += 1
            continue
        yaw = float(box[4]) if len(box) > 4 and np.isfinite(float(box[4])) else 0.0
        class_id = int(box[7])
        shell = _image_space_box_px(canvas, row, col, half_row, half_col, yaw, sx, sy)
        shadow = shell + np.array((2, 2), dtype=np.int32)
        cv2.polylines(canvas, [shadow], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(canvas, [shell], True, (116, 151, 171), 2, cv2.LINE_AA)
        drawn += 1
        if class_id == 1:
            cv2.circle(canvas, tuple(np.mean(shell, axis=0).astype(np.int32)), 5, (255, 82, 65), -1, cv2.LINE_AA)
    return drawn, skipped


def _image_space_box_px(
    canvas: np.ndarray,
    row: float,
    col: float,
    half_row: float,
    half_col: float,
    yaw: float,
    sx: float,
    sy: float,
) -> np.ndarray:
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )
    corners = np.array(
        [[-half_row, -half_col], [half_row, -half_col], [half_row, half_col], [-half_row, half_col]],
        dtype=np.float32,
    )
    rc = (rotation @ corners.T).T + np.array([row, col], dtype=np.float32)
    points = np.stack([rc[:, 1] * sx, rc[:, 0] * sy], axis=1)
    points[:, 0] = np.clip(points[:, 0], 0, canvas.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, canvas.shape[0] - 1)
    return np.round(points).astype(np.int32)


def _draw_bev_box_outline(
    canvas: np.ndarray,
    x_m: float,
    y_m: float,
    half_length: float,
    half_width: float,
    yaw: float,
    color: Tuple[int, int, int],
    fill: Optional[Tuple[int, int, int]] = None,
    thickness: int = 4,
) -> None:
    shell = _vehicle_profile_px(
        canvas, x_m, y_m, half_length, half_width, yaw,
        ((1.00, -1.00), (1.00, 1.00), (-1.00, 1.00), (-1.00, -1.00)),
    )
    if fill is not None:
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, shell, fill, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.36, canvas, 0.64, 0.0, dst=canvas)
    shadow = shell + np.array((2, 2), dtype=np.int32)
    cv2.polylines(canvas, [shadow], True, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.polylines(canvas, [shell], True, color, thickness, cv2.LINE_AA)


def _vehicle_profile_px(
    canvas: np.ndarray,
    x_m: float,
    y_m: float,
    half_length: float,
    half_width: float,
    yaw: float,
    profile: Iterable[Tuple[float, float]],
) -> np.ndarray:
    forward = np.array((math.cos(yaw), math.sin(yaw)), dtype=np.float32)
    right = np.array((-math.sin(yaw), math.cos(yaw)), dtype=np.float32)
    center = np.array((x_m, y_m), dtype=np.float32)
    points = [
        center + local_forward * half_length * forward + local_right * half_width * right
        for local_forward, local_right in profile
    ]
    return np.asarray(
        [_vehicle_to_bev_px(canvas, float(point[0]), float(point[1])) for point in points],
        dtype=np.int32,
    )


def _draw_scene_vehicle(
    canvas: np.ndarray,
    x_m: float,
    y_m: float,
    half_length: float,
    half_width: float,
    yaw: float,
    body: Tuple[int, int, int],
    ego: bool = False,
) -> None:
    shell = _vehicle_profile_px(
        canvas, x_m, y_m, half_length, half_width, yaw,
        ((1.00, -0.66), (0.78, -1.00), (-0.70, -1.00), (-1.00, -0.67),
         (-1.00, 0.67), (-0.70, 1.00), (0.78, 1.00), (1.00, 0.66)),
    )
    shadow = shell + np.array((3, 4), dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, shadow, (85, 87, 90), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.17, canvas, 0.83, 0.0, dst=canvas)
    cv2.fillConvexPoly(canvas, shell, body, cv2.LINE_AA)
    cv2.polylines(canvas, [shell], True, (105, 108, 111) if ego else (68, 72, 76), 1, cv2.LINE_AA)

    glass = _vehicle_profile_px(
        canvas, x_m, y_m, half_length, half_width, yaw,
        ((0.44, -0.66), (0.18, -0.72), (-0.36, -0.72), (-0.52, -0.55),
         (-0.52, 0.55), (-0.36, 0.72), (0.18, 0.72), (0.44, 0.66)),
    )
    cv2.fillConvexPoly(canvas, glass, (69, 76, 82) if not ego else (68, 76, 82), cv2.LINE_AA)
    divider = _vehicle_profile_px(
        canvas, x_m, y_m, half_length, half_width, yaw,
        ((0.08, -0.68), (0.08, 0.68)),
    )
    cv2.line(canvas, tuple(divider[0]), tuple(divider[1]), (183, 190, 194), 1, cv2.LINE_AA)
    lamps = _vehicle_profile_px(
        canvas, x_m, y_m, half_length, half_width, yaw,
        ((0.90, -0.50), (0.90, 0.50)),
    )
    for point in lamps:
        cv2.circle(canvas, tuple(point), 2, (244, 246, 235), -1, cv2.LINE_AA)
    rear_lamps = _vehicle_profile_px(
        canvas, x_m, y_m, half_length, half_width, yaw,
        ((-0.88, -0.51), (-0.88, 0.51)),
    )
    for point in rear_lamps:
        cv2.circle(canvas, tuple(point), 2, (198, 62, 51) if ego else (132, 57, 51), -1, cv2.LINE_AA)


def _draw_scene_pedestrian(canvas: np.ndarray, x_m: float, y_m: float) -> None:
    px, py = _vehicle_to_bev_px(canvas, x_m, y_m)
    color = (222, 188, 118)
    cv2.circle(canvas, (px, py - 5), 4, color, -1, cv2.LINE_AA)
    cv2.line(canvas, (px, py), (px, py + 10), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (px, py + 3), (px - 5, py + 8), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (px, py + 3), (px + 5, py + 8), color, 2, cv2.LINE_AA)


def _draw_ego(canvas: np.ndarray) -> None:
    shell = _vehicle_profile_px(
        canvas,
        0.0,
        0.0,
        EGO_HALF_LENGTH_M,
        EGO_HALF_WIDTH_M,
        0.0,
        ((1.0, -0.70), (1.0, 0.70), (-1.0, 0.70), (-1.0, -0.70)),
    )
    cv2.fillConvexPoly(canvas, shell, (245, 250, 252), cv2.LINE_AA)
    cv2.polylines(canvas, [shell], True, (183, 204, 215), 1, cv2.LINE_AA)
    px, py = _vehicle_to_bev_px(canvas, 0.0, 0.0)
    cv2.putText(
        canvas,
        "EGO",
        (px - 23, py + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 245, 248),
        2,
        cv2.LINE_AA,
    )


def _draw_bev_legend(ax: plt.Axes) -> None:
    entries = [
        (BLUE, "TF++ plan"),
        (ORANGE, "TTC gate zone"),
        ("#b9c8d5", "LiDAR"),
        (GREEN, "Ego"),
    ]
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.715, 0.025),
            0.235,
            0.245,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            transform=ax.transAxes,
            fc="#07131d",
            ec=EDGE_SOFT,
            lw=0.9,
            alpha=0.92,
            zorder=2,
        )
    )
    y = 0.22
    for color, label in entries:
        ax.plot([0.74, 0.79], [y, y], transform=ax.transAxes, color=color, lw=3, solid_capstyle="round", zorder=3)
        ax.text(0.81, y, label, transform=ax.transAxes, color=TEXT, fontsize=8.8, va="center", zorder=3)
        y -= 0.055


def _fmt_number(value: float, decimals: int, signed: bool = False) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    value = round(value, decimals)
    if value == 0:
        value = 0.0
    if signed and value != 0.0:
        return f"{value:+.{decimals}f}"
    return f"{value:.{decimals}f}"


def _draw_driving_state(ax: plt.Axes, fd: Dict[str, Any], hist: Dict[str, list]) -> None:
    _clear_panel(ax, facecolor="#04111f")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ego_speed = float(fd.get("ego_speed", 0.0))
    ego_kmh = ego_speed * 3.6
    intervention = bool(fd.get("intervention", False))

    for spine in ax.spines.values():
        spine.set_edgecolor("#0d5484")
        spine.set_linewidth(1.0)

    ax.text(0.5, 0.92, "DRIVING STATE", color="#edf5ff", fontsize=14.5,
            fontweight="bold", ha="center", va="center", transform=ax.transAxes)
    gauge_ax = ax.inset_axes([0.018, 0.035, 0.964, 0.845])
    gauge_ax.set_xlim(0, 1)
    gauge_ax.set_ylim(0, 1)
    gauge_ax.set_aspect("equal")
    gauge_ax.axis("off")
    _draw_speedometer(gauge_ax, ego_kmh, center=(0.5, 0.455), radius=0.448,
                      max_speed=240.0, active=intervention)


def _draw_speedometer(
    ax: plt.Axes,
    speed_kmh: float,
    center: Tuple[float, float],
    radius: float,
    max_speed: float = 240.0,
    active: bool = False,
) -> None:
    start_angle = 220.0
    end_angle = -40.0
    span = start_angle - end_angle
    band_width = 0.078
    progress = float(np.clip(speed_kmh / max_speed, 0.0, 1.0))

    ax.add_patch(mpatches.Wedge(center, radius, end_angle, start_angle,
                                width=band_width, facecolor="#06182f",
                                edgecolor="#0570c0", linewidth=0.85,
                                transform=ax.transAxes))
    segments = 14
    gap_frac = 0.06
    for idx in range(segments):
        seg_start = idx / segments
        seg_end = min((idx + 1.0 - gap_frac) / segments, 1.0)
        theta_high = start_angle - seg_start * span
        theta_low = start_angle - seg_end * span
        ax.add_patch(mpatches.Wedge(center, radius, theta_low, theta_high,
                                    width=band_width, facecolor="#092442" if idx % 2 else "#081e3a",
                                    edgecolor="#0a365c", linewidth=0.35, alpha=0.98,
                                    transform=ax.transAxes))

    if progress > 0.0:
        progress_end = start_angle - progress * span
        fill_color = "#17cfff" if active else "#079dff"
        for expansion, alpha in ((0.017, 0.10), (0.010, 0.18), (0.004, 0.30)):
            ax.add_patch(mpatches.Wedge(
                center, radius + expansion, progress_end, start_angle,
                width=band_width + expansion * 2.0, facecolor="#17bfff",
                edgecolor="none", alpha=alpha, transform=ax.transAxes,
            ))
        ax.add_patch(mpatches.Wedge(
            center, radius, progress_end, start_angle,
            width=band_width, facecolor=fill_color,
            edgecolor="#43dcff", linewidth=0.7, alpha=0.98,
            transform=ax.transAxes,
        ))
        theta = math.radians(progress_end)
        r_inner = radius - band_width - 0.004
        r_outer = radius + 0.004
        ax.plot([center[0] + math.cos(theta) * r_inner, center[0] + math.cos(theta) * r_outer],
                [center[1] + math.sin(theta) * r_inner, center[1] + math.sin(theta) * r_outer],
                color="#b9f4ff", lw=2.0, alpha=0.96, solid_capstyle="round",
                transform=ax.transAxes)

    for tick in range(0, int(max_speed) + 1, 40):
        theta = math.radians(start_angle - (tick / max_speed) * span)
        r_outer = radius + 0.002
        r_inner = radius - band_width - 0.002
        x0 = center[0] + math.cos(theta) * r_inner
        y0 = center[1] + math.sin(theta) * r_inner
        x1 = center[0] + math.cos(theta) * r_outer
        y1 = center[1] + math.sin(theta) * r_outer
        ax.plot([x0, x1], [y0, y1], color="#26ccff", lw=4.8, alpha=0.14,
                solid_capstyle="round", transform=ax.transAxes)
        ax.plot([x0, x1], [y0, y1], color="#83ecff", lw=1.55,
                solid_capstyle="round",
                transform=ax.transAxes)

    tick_outer = radius - band_width - 0.025
    for tick in range(0, int(max_speed) + 1, 5):
        theta = math.radians(start_angle - (tick / max_speed) * span)
        major = tick % 20 == 0
        tick_len = 0.038 if major else 0.022
        r_inner = tick_outer - tick_len
        x0 = center[0] + math.cos(theta) * r_inner
        y0 = center[1] + math.sin(theta) * r_inner
        x1 = center[0] + math.cos(theta) * tick_outer
        y1 = center[1] + math.sin(theta) * tick_outer
        ax.plot([x0, x1], [y0, y1], color="#f4f8ff",
                lw=2.1 if major else 1.0, alpha=0.98,
                solid_capstyle="butt", transform=ax.transAxes)

    for label, value in (("0", 0.0), (f"{int(max_speed)}", max_speed)):
        theta = math.radians(start_angle - (value / max_speed) * span)
        r_label = radius + 0.085
        ax.text(center[0] + r_label * math.cos(theta),
                center[1] + r_label * math.sin(theta) - 0.006,
                label, color="#afbdd4", fontsize=12.0,
                ha="center", va="center", transform=ax.transAxes)

    ax.text(center[0], center[1] + 0.025, _fmt_number(speed_kmh, 0), color="#f6faff",
            fontsize=49, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(center[0], center[1] - 0.12, "km/h", color="#c5d0db",
            fontsize=13.2, ha="center", va="center", transform=ax.transAxes)


def _draw_explainability(ax: plt.Axes, fd: Dict[str, Any], hist: Dict[str, Any]) -> None:
    _clear_panel(ax, facecolor="#03101d")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0d5484")
        spine.set_linewidth(1.0)

    ax.text(0.026, 0.914, "EXPLAINABILITY", color="#d9e6f5",
            fontsize=13.0, fontweight="bold", va="center", transform=ax.transAxes)

    margin = 0.025
    gap = 0.032
    total_w = 1.0 - margin * 2 - gap * 2
    weights = [0.31, 0.31, 0.38]
    widths = [total_w * weight / sum(weights) for weight in weights]
    xs = [margin]
    for width in widths[:-1]:
        xs.append(xs[-1] + width + gap)

    card_y = 0.075
    card_h = 0.73
    action_name = str(fd.get("action_name", "proceed"))
    action_label = _ACTION_LABELS.get(action_name, action_name)
    action_reason = _action_reason(fd, action_label)

    _draw_control_card(ax, xs[0], card_y, widths[0], card_h,
                       _dashboard_control_lines(fd))
    _draw_gate_card(ax, xs[1], card_y, widths[1], card_h,
                    _dashboard_gate_lines(fd))
    _draw_quote_card(ax, xs[2], card_y, widths[2], card_h, action_reason, hist)


def _dashboard_control_lines(fd: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    action_name = str(fd.get("action_name", "proceed"))
    action_label = _ACTION_LABELS.get(action_name, action_name)
    action_color = _META_COLORS.get(action_name, GREEN)
    multiplier = float(fd.get("multiplier", 1.0))
    steer = _optional_float(fd.get("steer"))
    throttle = _optional_float(fd.get("throttle"))
    brake = _optional_float(fd.get("brake"))

    steering = "n/a" if steer is None else f"{_fmt_number(steer * 70.0, 1, signed=True)} deg"
    throttle_text = "n/a" if throttle is None else f"{_fmt_number(throttle * 100.0, 0)} %"
    brake_text = "n/a" if brake is None else f"{_fmt_number(brake * 100.0, 0)} %"
    return [
        ("Action", action_label, action_color),
        ("Steering", steering, "#c0cad9"),
        ("Brake Cmd", brake_text, RED if brake and brake > 0.02 else "#c0cad9"),
        ("Throttle Cmd", throttle_text, GREEN if throttle and throttle > 0.02 else "#c0cad9"),
        ("Speed scale", f"x{_fmt_number(multiplier, 1)}", "#c0cad9"),
    ]


def _dashboard_gate_lines(fd: Dict[str, Any]) -> List[Tuple[str, str]]:
    ttc = float(fd.get("ttc", 999.0))
    threshold = float(fd.get("ttc_threshold", 1.5))
    action_name = str(fd.get("action_name", "proceed"))
    action_label = _ACTION_LABELS.get(action_name, action_name)
    multiplier = float(fd.get("multiplier", 1.0))
    before = float(fd.get("tfpp_speed_mps", float("nan")))
    after = float(fd.get("final_speed_mps", float("nan")))

    if ttc < 999.0:
        comparator = "<=" if ttc <= threshold else ">"
        ttc_line = f"TTC {_fmt_number(ttc, 2)}s {comparator} threshold {_fmt_number(threshold, 1)}s"
    else:
        ttc_line = f"TTC n/a > threshold {_fmt_number(threshold, 1)}s"

    if math.isfinite(before) and math.isfinite(after):
        speed_line = f"Target speed {_fmt_number(before * 3.6, 1)} -> {_fmt_number(after * 3.6, 1)} km/h"
    else:
        speed_line = "Target speed n/a"

    action_color = _META_COLORS.get(action_name, GREEN)
    return [
        (ttc_line, "#d3e3f7"),
        (f'VLA selected "{action_label}"', action_color),
        (f"Speed scale x{_fmt_number(multiplier, 1)}", "#d3e3f7"),
        (speed_line, "#d3e3f7"),
    ]


def _draw_control_card(ax: plt.Axes, x: float, y: float, w: float, h: float,
                       rows: List[Tuple[str, str, str]]) -> None:
    _rounded_axes_patch(ax, x, y, w, h, "#061223e6", "#0d78bc", lw=1.15)
    ax.text(x + w * 0.50, y + h * 0.86, "APPLIED CONTROL", color="#85bdff",
            fontsize=9.7, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)

    row_top = y + h * 0.66
    row_gap = h * 0.135
    for idx, (label, value, value_color) in enumerate(rows):
        row_y = row_top - idx * row_gap
        ax.text(x + w * 0.075, row_y, f"{label}:", color="#a9c4e4",
                fontsize=8.5, va="center", transform=ax.transAxes)
        ax.text(x + w * 0.925, row_y, value, color=value_color,
                fontsize=8.7, fontweight="bold", ha="right", va="center",
                transform=ax.transAxes)
        if idx < len(rows) - 1:
            ax.plot([x + w * 0.075, x + w * 0.925],
                    [row_y - row_gap * 0.52, row_y - row_gap * 0.52],
                    color="#193651", linewidth=0.7, alpha=0.8,
                    transform=ax.transAxes)


def _draw_gate_card(ax: plt.Axes, x: float, y: float, w: float, h: float,
                    lines: List[Tuple[str, str]]) -> None:
    _rounded_axes_patch(ax, x, y, w, h, "#061223e6", "#0d78bc", lw=1.15)
    ax.text(x + w * 0.50, y + h * 0.86, "GATE LOGIC", color="#85bdff",
            fontsize=9.7, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)

    line_top = y + h * 0.66
    line_gap = h * 0.15
    for idx, (line, color) in enumerate(lines):
        line_y = line_top - idx * line_gap
        ax.text(x + w * 0.085, line_y, line, color=color, fontsize=8.0,
                va="center", transform=ax.transAxes)
        if idx < len(lines) - 1:
            ax.plot([x + w * 0.085, x + w * 0.915],
                    [line_y - line_gap * 0.52, line_y - line_gap * 0.52],
                    color="#193651", linewidth=0.7, alpha=0.8,
                    transform=ax.transAxes)


def _draw_quote_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    explanation: str,
    hist: Dict[str, Any],
) -> None:
    _rounded_axes_patch(ax, x, y, w, h, "#061e1adf", "#43af5b", lw=1.2)
    ax.text(x + w * 0.50, y + h * 0.86, "DRIVING SITUATION EXPLANATION",
            color=GREEN, fontsize=8.6, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    wrap_width = max(30, int(108 * w))
    explanation = _compact_display_caption(explanation, max_chars=wrap_width * 4 - 2)
    visible_explanation = _held_quote_caption(explanation, hist)
    quote = f'"{visible_explanation}"' if visible_explanation else '""'
    quote_lines = textwrap.wrap(quote, width=wrap_width)[:4]
    line_top = y + h * 0.66
    line_gap = h * 0.14
    for idx, line in enumerate(quote_lines):
        ax.text(x + w * 0.09, line_top - idx * line_gap, line, color="#edf4f5",
                fontsize=9.0, va="center", transform=ax.transAxes)


def _held_quote_caption(explanation: str, hist: Dict[str, Any]) -> str:
    """Keep the full explanation stable while the rest of the dashboard advances."""
    try:
        hold_steps = int(os.environ.get("META_DASHBOARD_EXPLANATION_HOLD_STEPS", "20"))
    except ValueError:
        hold_steps = 20
    if hold_steps <= 0:
        return explanation

    steps = hist.get("step") or [0]
    current_step = int(steps[-1]) if steps else 0
    held_text = str(hist.get("quote_held_text", ""))
    last_update_step = int(hist.get("quote_last_update_step", -10**9))

    if not held_text or current_step - last_update_step >= hold_steps:
        hist["quote_held_text"] = explanation
        hist["quote_last_update_step"] = current_step
        return explanation

    return held_text


def _rounded_axes_patch(ax: plt.Axes, x: float, y: float, w: float, h: float,
                        face: str, edge: str, lw: float = 1.0, alpha: float = 1.0) -> None:
    green_accent = edge.lower() in {"#43af5b", "#1b7a38", "#2da44e"}
    glow_color = "#43af5b" if green_accent else "#078ff0"
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        facecolor="none",
        edgecolor=glow_color,
        linewidth=3.2,
        alpha=0.12,
        transform=ax.transAxes,
    ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
        transform=ax.transAxes,
    ))


def _optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _env_float(name: str, default: float) -> float:
    try:
        result = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bullet_block(
    ax: plt.Axes,
    x: float,
    y: float,
    title: str,
    items: Iterable[str],
    max_x: float,
    max_chars: int,
    separator_y: Optional[float] = None,
) -> None:
    ax.text(x, y, title + ":", color=TEXT, fontsize=9.5, fontweight="bold", transform=ax.transAxes)
    yy = y - 0.055
    for item in items:
        lines = textwrap.wrap(str(item), width=max_chars) or [""]
        ax.text(x + 0.03, yy, "- " + lines[0], color="#dce6ef", fontsize=8.6, transform=ax.transAxes)
        yy -= 0.048
        for line in lines[1:]:
            ax.text(x + 0.055, yy, line, color="#dce6ef", fontsize=8.6, transform=ax.transAxes)
            yy -= 0.048
    line_y = yy + 0.025 if separator_y is None else separator_y
    ax.plot([x, max_x], [line_y, line_y], color=EDGE_SOFT, lw=0.7, ls="--",
            transform=ax.transAxes)


def _quote_block(
    ax: plt.Axes,
    x: float,
    y: float,
    title: str,
    sentence: str,
    max_x: float,
    max_chars: int,
    separator_y: Optional[float] = None,
) -> None:
    ax.text(x, y, title + ":", color=TEXT, fontsize=9.5, fontweight="bold", transform=ax.transAxes)
    quote = _quoted_sentence(sentence)
    yy = y - 0.065
    for line in textwrap.wrap(quote, width=max_chars) or ['""']:
        ax.text(x + 0.03, yy, line, color="#f0f6fb", fontsize=10.0, fontweight="bold", transform=ax.transAxes)
        yy -= 0.058
    line_y = yy + 0.030 if separator_y is None else separator_y
    ax.plot([x, max_x], [line_y, line_y], color=EDGE_SOFT, lw=0.7, ls="--",
            transform=ax.transAxes)


def _quoted_sentence(sentence: str) -> str:
    clean = " ".join(str(sentence).strip().strip('"').split())
    if not clean:
        clean = "No driving situation explanation is available yet."
    return f'"{clean}"'


def _compact_display_caption(sentence: str, max_chars: int) -> str:
    clean = " ".join(str(sentence).strip().strip('"').split())
    if not clean:
        return "No driving situation explanation is available yet."
    clean = re.sub(r"^Front(?: view)?(?: shows)?\s*:?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"\s*;\s*(?:rear(?: view)?(?: shows)?\s*:?\s*)",
        "; ",
        clean,
        flags=re.IGNORECASE,
    )
    clean = clean[:1].upper() + clean[1:]
    if len(clean) <= max_chars:
        return clean
    for old, new in (
        ("beside the ego vehicle", "beside ego"),
        ("in foggy, wet conditions", "on a wet road"),
        ("in wet, foggy conditions", "on a wet road"),
        ("multiple vehicles", "vehicles"),
        ("several vehicles", "vehicles"),
        ("dark urban roadway", "urban roadway"),
        ("dark urban street", "urban street"),
        ("with limited visibility", "with low visibility"),
    ):
        clean = clean.replace(old, new)
    if len(clean) <= max_chars:
        return clean
    clipped = clean[: max(1, max_chars - 1)].rsplit(" ", 1)[0].rstrip(" ,;:.!?")
    return f"{clipped}."


def _action_reason(fd: Dict[str, Any], action_label: str) -> str:
    dashboard_reason = str(fd.get("dashboard_vlm_reason") or "").strip()
    if dashboard_reason:
        return dashboard_reason

    reason = str(fd.get("action_reason") or fd.get("vla_reason") or fd.get("reason") or "").strip()
    if reason and not _looks_internal_explanation(reason):
        return reason

    return _caption_reason_from_frame(fd, action_label)


def _looks_internal_explanation(reason: str) -> bool:
    lower = " ".join(str(reason).lower().split())
    internal_terms = (
        "threshold",
        "ttc",
        "safety override",
        "speed scale",
        "vla",
        "obstacle distance",
        "dashboard-only",
    )
    return any(term in lower for term in internal_terms)


def _caption_reason_from_frame(fd: Dict[str, Any], action_label: str) -> str:
    action_name = str(fd.get("action_name", "proceed"))
    ttc = float(fd.get("ttc", 999.0))
    threshold = float(fd.get("ttc_threshold", 1.5))
    cue_label, _ = _dominant_scene_cue(fd.get("pred_boxes"), action_name)

    if action_name == "stop":
        if cue_label == "red traffic light":
            return "The ego vehicle is approaching a red traffic light relevant to its lane, requiring it to stop."
        if cue_label == "stop sign":
            return "Stop sign is visible and relevant to ego lane, requiring stop."
        target = _cue_with_article(cue_label or "obstacle")
        return f"Ego corridor is blocked by {target} ahead, requiring it to stop."

    if action_name == "yield":
        target = _cue_with_article(cue_label or "road user")
        return f"{target.capitalize()} is entering the ego corridor, requiring the ego vehicle to yield."

    if action_name == "slow_down" or ttc < threshold:
        target = _cue_with_article(cue_label or "vehicle")
        return (
            f"Ego corridor is blocked by {target} ahead, requiring immediate speed reduction."
        )

    if action_name in {"turn_left", "turn_right", "change_lane_left", "change_lane_right"}:
        return f"The ego vehicle has a clear maneuver corridor, requiring {action_label.lower()}."

    return "No immediate collision risk is detected. Maintain the current plan safely."


def _dominant_scene_cue(boxes: Any, action_name: str) -> Tuple[Optional[str], Optional[float]]:
    arr = _to_numpy(boxes)
    if arr is None:
        return None, None
    try:
        arr = np.asarray(list(arr), dtype=np.float32)
    except Exception:
        return None, None
    if arr.ndim != 2 or arr.shape[1] < 8 or arr.size == 0:
        return None, None

    best_obstacle: Tuple[Optional[str], Optional[float]] = (None, None)
    best_signal: Tuple[Optional[str], Optional[float]] = (None, None)
    for box in arr:
        x_m = float(box[0])
        y_m = float(box[1])
        if x_m < 0.5:
            continue
        class_id = int(box[7])
        cue = _CLASS_CUE_LABELS.get(class_id)
        if cue is None:
            continue

        if class_id in (0, 1, 4) and abs(y_m) <= 2.4:
            if best_obstacle[1] is None or x_m < best_obstacle[1]:
                best_obstacle = (cue, x_m)
        elif class_id in (2, 3) and abs(y_m) <= 5.0:
            if best_signal[1] is None or x_m < best_signal[1]:
                best_signal = (cue, x_m)

    if action_name == "stop" and best_signal[0] is not None:
        return best_signal
    if best_obstacle[0] is not None:
        return best_obstacle
    return best_signal


def _cue_with_article(cue_label: str) -> str:
    cue = str(cue_label).strip()
    if not cue:
        cue = "obstacle"
    article = "an" if cue[0].lower() in "aeiou" else "a"
    return f"{article} {cue}"


def _clear_panel(ax: plt.Axes, facecolor: str = PANEL, border: bool = True) -> None:
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(EDGE)
            spine.set_linewidth(0.9)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)


def _panel_title(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.020,
        0.965,
        title,
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.10", fc="#06111acc", ec=EDGE_SOFT, lw=0.9),
        zorder=10,
    )


def _image_from_frame(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    img = np.asarray(value)
    if img.ndim != 3 or img.shape[2] < 3:
        return None
    img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _remove_front_green_markers(img: np.ndarray) -> np.ndarray:
    """Remove small synthetic green waypoint dots from the raw front camera panel."""
    if img.size == 0:
        return img

    rgb = np.asarray(img)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    bright_green = (
        (g > 135)
        & (g - r > 55)
        & (g - b > 55)
        & (r < 120)
        & (b < 120)
    )
    dim_green = (
        (g > 38)
        & (g - r > 8)
        & (g - b > 8)
        & (r < 0.95 * g)
        & (b < 0.98 * g)
    )
    mask = (bright_green | dim_green).astype(np.uint8)
    if int(mask.sum()) == 0:
        return img

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros(mask.shape, dtype=np.uint8)
    h, w = mask.shape
    for label in range(1, n_labels):
        x, y, comp_w, comp_h, area = stats[label]
        if area > 3500 or comp_w > 90 or comp_h > 90:
            continue
        if y < int(h * 0.18):
            continue
        clean_mask[labels == label] = 255

    if int(clean_mask.sum()) == 0:
        return img
    clean_mask = cv2.dilate(clean_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return cv2.inpaint(rgb.copy(), clean_mask, 5, cv2.INPAINT_TELEA)


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_points(value: Any) -> Optional[np.ndarray]:
    arr = _to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and arr.size >= 4 and arr.size % 2 == 0:
        arr = arr.reshape(arr.size // 2, 2)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return None
    return arr[:, :2]


def _bev_center(canvas: np.ndarray) -> Tuple[int, int]:
    return int(canvas.shape[1] * 0.50), int(canvas.shape[0] * 0.75)


def _bev_scale(canvas: np.ndarray) -> float:
    return min(canvas.shape[0], canvas.shape[1]) / 68.0


def _vehicle_to_bev_px(canvas: np.ndarray, x_m: float, y_m: float) -> Tuple[int, int]:
    center = _bev_center(canvas)
    scale = _bev_scale(canvas)
    px = int(round(center[0] + y_m * scale))
    py = int(round(center[1] - x_m * scale))
    return px, py
