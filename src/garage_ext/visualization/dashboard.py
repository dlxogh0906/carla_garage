"""2x2 driving dashboard visualization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_HISTORY = {}


def visualize_frame(frame_data, save_path: Optional[str] = None):
  """Render a 2x2 dashboard and optionally save it to disk.

  Required keys:
    image, tfpp_waypoints, final_waypoints, danger_score, vlm_json

  Optional keys:
    vlm_waypoints, pred_boxes, history_key, reset_history
  """
  image = _ensure_rgb(frame_data["image"])
  tfpp_waypoints = _normalize_waypoints(frame_data.get("tfpp_waypoints"))
  vlm_waypoints = _normalize_waypoints(frame_data.get("vlm_waypoints"))
  final_waypoints = _normalize_waypoints(frame_data.get("final_waypoints"))
  if final_waypoints is None:
    final_waypoints = tfpp_waypoints
  danger_score = float(frame_data.get("danger_score", 0.0))
  raw_danger_score = float(frame_data.get("raw_danger_score", danger_score))
  vlm_json = frame_data.get("vlm_json") or {}
  pred_boxes = frame_data.get("pred_boxes")
  history_key = frame_data.get("history_key", "default")
  show_debug_vlm = _show_debug_vlm(frame_data)
  history = _get_history(history_key, reset=bool(frame_data.get("reset_history", False)))
  intervention = bool(frame_data.get("intervention", danger_score >= 0.5))
  mode = frame_data.get("mode") or ("VLM intervention" if intervention else "TF++ active")
  l2_distance = _waypoint_l2(tfpp_waypoints, vlm_waypoints)
  history["danger"].append(danger_score)
  history["raw_danger"].append(raw_danger_score)
  history["l2"].append(l2_distance)
  history["count"] += int(intervention)

  camera_panel = _draw_camera_panel(image, tfpp_waypoints, vlm_waypoints, final_waypoints, danger_score, vlm_json, mode)
  bev_panel = _draw_bev_panel(tfpp_waypoints, vlm_waypoints, final_waypoints, pred_boxes)

  fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
  axes[0, 0].imshow(camera_panel)
  axes[0, 0].set_title("Camera View", fontsize=12, weight="bold")
  axes[0, 1].imshow(bev_panel)
  axes[0, 1].set_title("Bird-Eye View", fontsize=12, weight="bold")
  axes[0, 0].axis("off")
  axes[0, 1].axis("off")

  _draw_vlm_panel(
      axes[1, 0],
      danger_score,
      raw_danger_score,
      vlm_json,
      intervention,
      mode=mode,
      vlm_ran=bool(frame_data.get("vlm_ran", False)),
      parse_error=frame_data.get("parse_error"),
      vlm_suggested_action=frame_data.get("vlm_suggested_action"),
      final_selected_action=frame_data.get("final_selected_action"),
      danger_components=frame_data.get("danger_components") or {},
      tfpp_speed_mps=frame_data.get("tfpp_speed_mps"),
      vlm_speed_mps=frame_data.get("vlm_speed_mps"),
      final_speed_mps=frame_data.get("final_speed_mps"),
      show_debug_vlm=show_debug_vlm,
  )
  _draw_metrics_panel(axes[1, 1], history, show_debug_vlm=show_debug_vlm)

  fig.tight_layout()
  fig.canvas.draw()
  width, height = fig.canvas.get_width_height()
  rendered = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
  plt.close(fig)

  if save_path is not None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

  return rendered


def _get_history(key: str, reset: bool = False) -> dict:
  if reset or key not in _HISTORY:
    _HISTORY[key] = {"danger": [], "raw_danger": [], "l2": [], "count": 0}
  return _HISTORY[key]


def _show_debug_vlm(frame_data) -> bool:
  if "show_raw_vlm" in frame_data:
    return bool(frame_data.get("show_raw_vlm"))
  return os.environ.get("VLA_VIS_DEBUG", "0") == "1"


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
  image = np.asarray(image)
  if image.ndim != 3 or image.shape[2] != 3:
    raise ValueError(f"Expected an RGB image shaped (H, W, 3), got {image.shape}.")
  if image.dtype != np.uint8:
    image = np.clip(image, 0, 255).astype(np.uint8)
  return image


def _normalize_waypoints(waypoints):
  if waypoints is None:
    return None
  array = np.asarray(waypoints, dtype=np.float32)
  if array.ndim == 3 and array.shape[0] == 1:
    array = array[0]
  if array.ndim == 1 and array.size == 20:
    array = array.reshape(10, 2)
  if array.shape != (10, 2):
    return None
  return array


def _draw_camera_panel(image, tfpp_waypoints, vlm_waypoints, final_waypoints, danger_score, vlm_json, mode_text):
  canvas = image.copy()
  overlay = canvas.copy()

  if vlm_json.get("red_light_applies_to_ego", vlm_json.get("red_light")):
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (220, 40, 40), -1)
  if vlm_json.get("pedestrian_in_path") or vlm_json.get("pedestrian"):
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (255, 140, 0), -1)
  if vlm_json.get("emergency_vehicle"):
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (70, 130, 220), -1)
  if danger_score >= 0.5:
    canvas = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0.0)

  _draw_projected_waypoints(canvas, tfpp_waypoints, (60, 120, 255), dotted=True, thickness=2)
  _draw_projected_waypoints(canvas, vlm_waypoints, (235, 70, 70), dotted=True, thickness=2)
  _draw_projected_waypoints(canvas, final_waypoints, (40, 180, 70), dotted=False, thickness=4)

  cv2.putText(canvas, mode_text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
  return canvas


def _draw_projected_waypoints(canvas, waypoints, color, dotted: bool, thickness: int) -> None:
  if waypoints is None:
    return
  points = _project_to_image(waypoints, canvas.shape[:2])
  if len(points) < 2:
    return
  _draw_polyline(canvas, points, color=color, dotted=dotted, thickness=thickness)


def _project_to_image(waypoints: np.ndarray, image_shape) -> list:
  height, width = image_shape
  points = []
  for x_meters, y_meters in waypoints:
    x_meters = max(float(x_meters), 0.0)
    depth = max(1.0, 1.0 + 0.18 * x_meters)
    px = width * 0.5 + (y_meters * width * 0.09) / depth
    py = height * 0.92 - (x_meters * height * 0.07) / depth
    points.append((int(round(px)), int(round(py))))
  return points


def _draw_polyline(canvas, points, color, dotted: bool, thickness: int) -> None:
  if not dotted:
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
    for point in points:
      cv2.circle(canvas, point, max(2, thickness), color, -1, cv2.LINE_AA)
    return

  for start, end in zip(points[:-1], points[1:]):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    segment = end - start
    distance = float(np.linalg.norm(segment))
    if distance < 1e-6:
      continue
    direction = segment / distance
    dash = 8.0
    gap = 6.0
    position = 0.0
    while position < distance:
      dash_start = start + direction * position
      dash_end = start + direction * min(position + dash, distance)
      cv2.line(canvas, tuple(dash_start.astype(int)), tuple(dash_end.astype(int)), color, thickness, cv2.LINE_AA)
      position += dash + gap


def _draw_bev_panel(tfpp_waypoints, vlm_waypoints, final_waypoints, pred_boxes):
  canvas = np.full((620, 620, 3), 245, dtype=np.uint8)
  center = np.array([canvas.shape[1] // 2, int(canvas.shape[0] * 0.82)])
  pixels_per_meter = 18.0

  for offset in range(-8, 9):
    x = int(center[0] + offset * pixels_per_meter * 2)
    y = int(center[1] - offset * pixels_per_meter * 2)
    cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, y), (canvas.shape[1], y), (220, 220, 220), 1, cv2.LINE_AA)

  cv2.rectangle(canvas, (center[0] - 16, center[1] - 28), (center[0] + 16, center[1] + 28), (35, 45, 55), -1)
  cv2.putText(canvas, "EGO", (center[0] - 18, center[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

  _draw_bev_waypoints(canvas, tfpp_waypoints, center, pixels_per_meter, (60, 120, 255), dotted=True, thickness=2)
  _draw_bev_waypoints(canvas, vlm_waypoints, center, pixels_per_meter, (235, 70, 70), dotted=True, thickness=2)
  _draw_bev_waypoints(canvas, final_waypoints, center, pixels_per_meter, (40, 180, 70), dotted=False, thickness=4)
  _draw_actor_marks(canvas, pred_boxes, center, pixels_per_meter)

  return canvas


def _draw_bev_waypoints(canvas, waypoints, center, pixels_per_meter, color, dotted, thickness):
  if waypoints is None:
    return

  points = []
  for x_meters, y_meters in waypoints:
    px = int(round(center[0] + y_meters * pixels_per_meter))
    py = int(round(center[1] - x_meters * pixels_per_meter))
    points.append((px, py))
  if len(points) < 2:
    return
  _draw_polyline(canvas, points, color=color, dotted=dotted, thickness=thickness)


def _draw_actor_marks(canvas, pred_boxes, center, pixels_per_meter):
  if pred_boxes is None:
    return

  boxes = np.asarray(pred_boxes)
  if boxes.ndim != 2 or boxes.shape[1] < 8:
    return

  for box in boxes:
    x_meters = float(box[0])
    y_meters = float(box[1])
    class_id = int(box[7])
    px = int(round(center[0] + y_meters * pixels_per_meter))
    py = int(round(center[1] - x_meters * pixels_per_meter))

    if class_id == 1:
      color = (255, 140, 0)
      radius = 7
    elif class_id == 0:
      color = (80, 90, 110)
      radius = 9
    elif class_id == 2:
      color = (220, 50, 50)
      radius = 8
    else:
      color = (120, 120, 120)
      radius = 6

    cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)


def _draw_vlm_panel(ax, danger_score, raw_danger_score, vlm_json, intervention, mode, vlm_ran=False, parse_error=None,
                    vlm_suggested_action=None, final_selected_action=None, danger_components=None,
                    tfpp_speed_mps=None, vlm_speed_mps=None, final_speed_mps=None, show_debug_vlm=False):
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis("off")

  bar_color = "#d94c4c" if intervention else "#3f7fba"
  if show_debug_vlm:
    ax.barh([0.82], [danger_score], height=0.10, color=bar_color)
    ax.barh([0.66], [raw_danger_score], height=0.08, color="#f0a34a")
    ax.barh([0.82], [1.0], height=0.10, color="none", edgecolor="#555555")
    ax.barh([0.66], [1.0], height=0.08, color="none", edgecolor="#999999")
    ax.text(0.0, 0.92, "danger_score (calibrated)", fontsize=11, weight="bold")
    ax.text(min(0.98, danger_score + 0.02), 0.82, f"{danger_score:.2f}", va="center", fontsize=10)
    ax.text(0.0, 0.73, "danger_score (raw VLM)", fontsize=10)
    ax.text(min(0.98, raw_danger_score + 0.02), 0.66, f"{raw_danger_score:.2f}", va="center", fontsize=10)
    text_y = 0.48
  else:
    ax.barh([0.80], [danger_score], height=0.10, color=bar_color)
    ax.barh([0.80], [1.0], height=0.10, color="none", edgecolor="#555555")
    ax.text(0.0, 0.90, "danger_score", fontsize=11, weight="bold")
    ax.text(min(0.98, danger_score + 0.02), 0.80, f"{danger_score:.2f}", va="center", fontsize=10)
    text_y = 0.62

  tags = []
  if vlm_json.get("red_light_applies_to_ego", vlm_json.get("red_light")):
    tags.append("red_light")
  elif vlm_json.get("red_light_visible"):
    tags.append("signal_visible")
  if vlm_json.get("pedestrian_in_path") or vlm_json.get("pedestrian"):
    tags.append("pedestrian")
  if vlm_json.get("vehicle_collision_risk") or vlm_json.get("collision_risk"):
    tags.append("collision_risk")
  if vlm_json.get("lead_vehicle"):
    tags.append("lead_vehicle")
  if vlm_json.get("cut_in_vehicle"):
    tags.append("cut_in_vehicle")
  if vlm_json.get("emergency_vehicle"):
    tags.append("emergency_vehicle")
  if not tags:
    tags.append("none")

  action = vlm_suggested_action or vlm_json.get("recommended_action")
  if action is None:
    action_speed = None
    action_block = vlm_json.get("action")
    if isinstance(action_block, dict):
      action_speed = action_block.get("speed")
    if action_speed is None:
      action = "unknown"
    elif float(action_speed) <= 0.01:
      action = "stop"
    elif danger_score >= 0.5:
      action = "slow"
    else:
      action = "proceed"

  if final_selected_action is None:
    final_selected_action = action

  component_text = None
  if danger_components:
    important = [(key, value) for key, value in danger_components.items() if float(value) > 0.0]
    if important:
      component_text = ", ".join(f"{key}:{value:.2f}" for key, value in important[:3])

  lines = [
      f"Scene tags: {', '.join(tags)}",
      f"VLM suggested action: {action}",
      f"Final selected action: {final_selected_action}",
      f"Current mode: {mode}",
      f"VLM ran this frame: {'yes' if vlm_ran else 'no'}",
  ]
  if component_text is not None:
    lines.append(f"Evidence: {component_text}")
  if "signal_confidence" in vlm_json:
    lines.append(f"Signal confidence: {float(vlm_json.get('signal_confidence', 0.0)):.2f}")
  if tfpp_speed_mps is not None:
    lines.append(f"TF++ speed: {float(tfpp_speed_mps):.2f} m/s")
  if vlm_speed_mps is not None:
    lines.append(f"VLM speed cap: {float(vlm_speed_mps):.2f} m/s")
  if final_speed_mps is not None:
    lines.append(f"Final speed: {float(final_speed_mps):.2f} m/s")
  if "collision_guard" in vlm_json:
    lines.append(f"Collision guard: {vlm_json['collision_guard'].get('reason', 'active')}")
  if "lane_change_guard" in vlm_json:
    guard = vlm_json["lane_change_guard"]
    lines.append(
        f"Lane-change guard: {guard.get('direction', '?')} {guard.get('nearest_distance_m', '?')}m"
    )
  if "red_light_guard" in vlm_json:
    lines.append(f"Signal guard: {vlm_json['red_light_guard'].get('type', 'active')}")
  if parse_error or "parse_error" in vlm_json:
    lines.append(f"Parser: {parse_error or vlm_json['parse_error']}")

  for idx, line in enumerate(lines):
    ax.text(0.0, text_y - idx * 0.10, line, fontsize=10.5)


def _draw_metrics_panel(ax, history, show_debug_vlm=False):
  ax.set_title("Metrics", fontsize=12, weight="bold")
  steps = np.arange(len(history["danger"]))
  ax.plot(steps, history["danger"], color="#d94c4c", linewidth=2, label="danger_score")
  if show_debug_vlm and history["raw_danger"]:
    ax.plot(steps, history["raw_danger"], color="#f0a34a", linewidth=1.8, linestyle="--", label="raw VLM danger")
  ax.set_ylim(0.0, 1.05)
  ax.set_xlabel("frame")
  ax.set_ylabel("score / distance")

  if history["l2"]:
    ax.plot(steps, history["l2"], color="#235a95", linewidth=2, label="TF++ vs Alpamayo L2")

  ax.legend(loc="upper left")
  ax.grid(True, alpha=0.25)
  ax.text(
      0.98,
      0.90,
      f"Alpamayo interventions: {history['count']}",
      ha="right",
      va="center",
      transform=ax.transAxes,
      fontsize=11,
      bbox={"boxstyle": "round", "facecolor": "#f4f4f4", "edgecolor": "#bbbbbb"},
  )


def _waypoint_l2(tfpp_waypoints, vlm_waypoints) -> float:
  if tfpp_waypoints is None or vlm_waypoints is None:
    return 0.0
  return float(np.linalg.norm(tfpp_waypoints - vlm_waypoints))
