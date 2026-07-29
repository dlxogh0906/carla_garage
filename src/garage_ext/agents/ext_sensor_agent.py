"""Extension agent: subclass of upstream SensorAgent that runs our pipeline.

Leaderboard loads an agent by importing the module named in its CLI
argument and calling `get_entry_point()`. We expose ExtSensorAgent here
so evaluations can point at this file instead of `team_code/sensor_agent.py`
without touching upstream code.

Key design points:
- We do NOT re-implement perception or planning. The parent class still
  does its thing; we only run ExtPipeline on top of its output.
- The YAML path is passed via env var GARAGE_EXT_CONFIG so we don't need
  to change leaderboard's CLI.
- If GARAGE_EXT_CONFIG is unset, the agent behaves exactly like upstream.
"""
import logging
import os
import pathlib
import re
from typing import Any, Optional

import cv2
import numpy as np

from sensor_agent import SensorAgent  # upstream, in team_code/

from garage_ext.config.ext_config import ExtConfig, load_experiment_config
from garage_ext.modules import base as mb
from garage_ext.modules import vlm, risk, safety, image_enhancer  # noqa: F401
from garage_ext.pipeline import ExtPipeline
from garage_ext.registry import build as registry_build
from garage_ext.visualization.dashboard import visualize_frame

_log = logging.getLogger(__name__)


def get_entry_point():
  return "ExtSensorAgent"


class ExtSensorAgent(SensorAgent):
  """Upstream SensorAgent + team modules (image enhancer / VLM / risk / safety)."""

  _route_setup_count = 0  # class-level counter, incremented each setup() call

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    ExtSensorAgent._route_setup_count += 1
    self.save_path = None  # ensure attribute exists even if super().setup() raises
    super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)

    # VIZ_ROUTE_LIMIT=N: save debug images only for the first N routes.
    # Routes beyond the limit have save_path nulled so visualize_model() is skipped.
    viz_limit = int(os.environ.get("VIZ_ROUTE_LIMIT", "0"))
    if viz_limit > 0 and ExtSensorAgent._route_setup_count > viz_limit:
      self.save_path = None
    self._tf_pred_checkpoints = None
    self._tf_pred_target_speed_mps = None
    self._tf_direct_capture_patched = False
    self._ext_cfg = self._load_ext_config()
    self._ext_pipeline_enabled = self._needs_ext_pipeline(self._ext_cfg)
    if self._ext_pipeline_enabled:
      self._patch_direct_controller_capture()
    self._ext_pipeline = ExtPipeline(self._ext_cfg) if self._ext_pipeline_enabled else None
    self._image_enhancer = (registry_build("image_enhancer", self._ext_cfg.image_enhancer, **
                                           self._ext_cfg.image_enhancer_kwargs)
                            if self._ext_cfg and self._ext_cfg.image_enhancer else None)
    self._save_interval = 4
    self._compare_counter = 0
    self._compare_root = (pathlib.Path(os.environ.get("SAVE_PATH", "")) if os.environ.get("SAVE_PATH") else None)
    self._route_tag = self._extract_route_tag(path_to_conf_file)
    self._compare_save_path = self._build_compare_save_path()
    self._dashboard_interval = max(1, int(os.environ.get("EXT_DASHBOARD_INTERVAL", "1")))
    self._enhance_compare_rear = os.environ.get("EXT_ENHANCE_COMPARE_REAR", "0") == "1"

  def sensors(self):
    sensors = super().sensors()
    if os.environ.get("EXT_ENHANCE_COMPARE_REAR", "0") != "1":
      return sensors
    if any(sensor.get("id") == "rgb_rear" for sensor in sensors):
      return sensors

    def env_float(name: str, default: float) -> float:
      try:
        return float(os.environ.get(name, default))
      except (TypeError, ValueError):
        return default

    sensors.append({
        "type": "sensor.camera.rgb",
        "x": env_float("EXT_REAR_CAMERA_X", float(self.config.camera_pos[0])),
        "y": env_float("EXT_REAR_CAMERA_Y", float(self.config.camera_pos[1])),
        "z": env_float("EXT_REAR_CAMERA_Z", float(self.config.camera_pos[2])),
        "roll": env_float("EXT_REAR_CAMERA_ROLL", float(self.config.camera_rot_0[0])),
        "pitch": env_float("EXT_REAR_CAMERA_PITCH", float(self.config.camera_rot_0[1])),
        "yaw": env_float("EXT_REAR_CAMERA_YAW", float(self.config.camera_rot_0[2]) + 180.0),
        "width": int(env_float("EXT_REAR_CAMERA_WIDTH", float(self.config.camera_width))),
        "height": int(env_float("EXT_REAR_CAMERA_HEIGHT", float(self.config.camera_height))),
        "fov": env_float("EXT_REAR_CAMERA_FOV", float(self.config.camera_fov)),
        "id": "rgb_rear",
    })
    return sensors

  def _patch_direct_controller_capture(self) -> None:
    """Capture TF++ direct-controller predictions without editing upstream code."""
    if getattr(self, "_tf_direct_capture_patched", False):
      return

    nets = getattr(self, "nets", None)
    if not nets:
      return

    net = nets[0]
    original = getattr(net, "control_pid_direct", None)
    if original is None or getattr(net, "_garage_ext_capture_wrapped", False):
      return

    agent_self = self

    def capture_wrapper(pred_checkpoints, pred_target_speed, speed, *args, **kwargs):
      agent_self._capture_tf_direct_predictions(pred_checkpoints, pred_target_speed)
      return original(pred_checkpoints, pred_target_speed, speed, *args, **kwargs)

    setattr(net, "_garage_ext_control_pid_direct_original", original)
    setattr(net, "control_pid_direct", capture_wrapper)
    setattr(net, "_garage_ext_capture_wrapped", True)
    self._tf_direct_capture_patched = True

  @staticmethod
  def _needs_ext_pipeline(cfg: ExtConfig | None) -> bool:
    """Image-only configs should not run ExtPipeline."""
    if cfg is None:
      return False
    return bool(
        cfg.vlm
        or (cfg.risk and cfg.risk != "noop")
        or (cfg.safety and cfg.safety != "noop")
    )

  @staticmethod
  def _as_numpy(value):
    if value is None:
      return None
    try:
      if hasattr(value, "detach"):
        value = value.detach()
      if hasattr(value, "cpu"):
        value = value.cpu()
      if hasattr(value, "numpy"):
        value = value.numpy()
      return np.asarray(value, dtype=np.float32)
    except Exception:
      return None

  @classmethod
  def _as_float(cls, value):
    arr = cls._as_numpy(value)
    if arr is not None:
      try:
        return float(arr.reshape(-1)[0])
      except Exception:
        pass
    try:
      return float(value)
    except Exception:
      return None

  def _capture_tf_direct_predictions(self, pred_checkpoints, pred_target_speed) -> None:
    checkpoints = self._as_numpy(pred_checkpoints)
    if checkpoints is not None:
      if checkpoints.ndim == 3 and checkpoints.shape[0] == 1:
        checkpoints = checkpoints[0]
      if checkpoints.ndim == 2 and checkpoints.shape[1] >= 2:
        self._tf_pred_checkpoints = checkpoints[:, :2].tolist()

    target_speed = self._as_float(pred_target_speed)
    if target_speed is not None:
      self._tf_pred_target_speed_mps = target_speed

  def _get_tf_prediction_context(self) -> dict:
    tf_pred_wp = None
    if hasattr(self, "pred_wp") and self.pred_wp is not None:
      try:
        tf_pred_wp = self.pred_wp[0].detach().cpu().numpy().tolist()  # (8, 2) float
      except Exception:
        pass

    tf_pred_checkpoints = getattr(self, "_tf_pred_checkpoints", None)
    tf_pred_target_speed = getattr(self, "_tf_pred_target_speed_mps", None)

    if tf_pred_wp is not None:
      tf_pred_path = tf_pred_wp
      tf_pred_path_kind = "wp_gru"
    elif tf_pred_checkpoints is not None:
      tf_pred_path = tf_pred_checkpoints
      tf_pred_path_kind = "checkpoint"
    else:
      tf_pred_path = None
      tf_pred_path_kind = None

    return {
        "tf_pred_wp": tf_pred_wp,
        "tf_pred_checkpoints": tf_pred_checkpoints,
        "tf_pred_target_speed_mps": tf_pred_target_speed,
        "tf_pred_path": tf_pred_path,
        "tf_pred_path_kind": tf_pred_path_kind,
    }

  @staticmethod
  def _load_ext_config() -> ExtConfig | None:
    cfg_path = os.environ.get("GARAGE_EXT_CONFIG")
    if not cfg_path:
      return None
    return load_experiment_config(cfg_path)

  def _extract_route_tag(self, path_to_conf_file: str) -> str:
    """Use Bench2Drive's agent-config suffix to recover the actual route id."""
    path_parts = path_to_conf_file.split("+")
    if len(path_parts) <= 1:
      return "unknown_route"

    save_name = path_parts[-1]
    match = re.search(r"(RouteScenario_\d+_rep\d+)", save_name)
    if match:
      return match.group(1)
    return save_name or "unknown_route"

  def _build_compare_save_path(self) -> pathlib.Path | None:
    root = getattr(self, "_compare_root", None)
    if root is None:
      return None
    route_tag = getattr(self, "_route_tag", "unknown_route")
    return root / route_tag / "enhance_compare"

  def _apply_image_enhancement(self, input_data: dict) -> dict:
    """Return a shallow-copied input_data with enhanced camera frames."""
    if self._image_enhancer is None:
      return input_data
    enhanced = dict(input_data)
    compare_frames = []
    for key in list(enhanced.keys()):
      if not key.startswith("rgb_"):
        continue
      sensor_id, frame = enhanced[key]
      bgr = frame[:, :, :3].copy()
      bgr_enh = self._image_enhancer.enhance(bgr)
      new_frame = frame.copy()
      new_frame[:, :, :3] = bgr_enh
      enhanced[key] = (sensor_id, new_frame)
      compare_frames.append((key, bgr, bgr_enh))

    self._save_compare(compare_frames)
    return enhanced

  def _save_compare(self, frames: list) -> None:
    """원본 | 보정 비교 이미지를 저장. SAVE_PATH 없으면 스킵."""
    if self._compare_save_path is None or not frames:
      return
    self._compare_counter = getattr(self, '_compare_counter', 0) + 1
    if self._compare_counter % self._save_interval != 0:
      return
    order = {"rgb_front": 0, "rgb_rear": 1}
    panels = []
    for _, bgr_orig, bgr_enh in sorted(frames, key=lambda item: order.get(item[0], 99)):
      panels.append(np.concatenate([bgr_orig, bgr_enh], axis=1))

    self._compare_save_path.mkdir(parents=True, exist_ok=True)
    comparison = np.concatenate(panels, axis=0)
    cv2.imwrite(str(self._compare_save_path / f"{self._compare_counter:05d}.png"), comparison)

  def run_step(self, input_data, timestamp, sensors=None, plan=None):
    if self._ext_pipeline is not None:
      self._patch_direct_controller_capture()

    # ★ 이미지 보정: upstream 모델이 보기 전에 적용
    input_data = self._apply_image_enhancement(input_data)

    control = super().run_step(input_data, timestamp, sensors=sensors)
    if self._ext_pipeline is None:
      return control

    # Wrap upstream state into our neutral dataclasses. We deliberately
    # keep this thin: modules that need more can read self (or upstream
    # attrs) via the `obs.data` bag.
    obs = mb.Observation(data={
        "input_data": input_data,
        "timestamp": timestamp,
        "agent": self,
        **self._get_tf_prediction_context(),
    })
    plan_obj = mb.Plan(meta={"source": "upstream_sensor_agent"})
    base_control = mb.Control(
        steer=float(getattr(control, "steer", 0.0)),
        throttle=float(getattr(control, "throttle", 0.0)),
        brake=float(getattr(control, "brake", 0.0)),
    )

    out = self._ext_pipeline.run(obs, plan_obj, base_control)
    control.steer = out.control.steer
    control.throttle = out.control.throttle
    control.brake = out.control.brake
    self._save_dashboard_frame(input_data, out, base_control)
    return control

  # ------------------------------------------------------------------
  # Dashboard visualization
  # ------------------------------------------------------------------

  def _dashboard_save_root(self) -> Optional[pathlib.Path]:
    """Return the per-route save root, falling back to SAVE_PATH+route_tag when
    sensor_agent nulled self.save_path (happens when route_index is None)."""
    if self.save_path is not None:
      return pathlib.Path(self.save_path)
    root = getattr(self, "_compare_root", None)
    if root is None:
      return None
    return root / getattr(self, "_route_tag", "unknown_route")

  def _save_dashboard_frame(self, input_data: dict, out, base_control) -> None:
    save_root = self._dashboard_save_root()
    if save_root is None:
      return
    if self.step % self._dashboard_interval != 0:
      return

    try:
      bgr = input_data["rgb_front"][1][:, :, :3]
      rgb = bgr[:, :, ::-1].copy()
    except Exception:
      return

    hint = (out.vlm_info.get("vlm_hint") or {}) if out.vlm_info else {}
    pred_xyz = hint.get("pred_xyz")
    vlm_json = self._build_dashboard_vlm_json(hint, out.risk)

    frame_data = {
        "image": rgb,
        "tfpp_waypoints": self._tf_pred_checkpoints,
        "vlm_waypoints": self._resample_alp_waypoints(pred_xyz),
        "final_waypoints": self._tf_pred_checkpoints,
        "danger_score": float(out.risk.score),
        "vlm_json": vlm_json,
        "pred_boxes": self.bb_buffer[-1] if len(self.bb_buffer) > 0 else None,
        "history_key": self._route_tag,
        "reset_history": self.step <= 1,
        "mode": self._dashboard_mode(out.control),
        "intervention": bool(out.control.meta.get("semantic_arbiter", False)),
        "vlm_ran": bool(hint),
        "vlm_suggested_action": vlm_json.get("recommended_action"),
        "final_selected_action": self._dashboard_final_action(out.control, base_control),
        "tfpp_speed_mps": self._tf_pred_target_speed_mps,
        "vlm_speed_mps": hint.get("target_speed_cap_mps"),
        "final_speed_mps": self._tf_pred_target_speed_mps,
    }

    save_path = save_root / "dashboard" / f"{self.step:05d}.png"
    try:
      visualize_frame(frame_data, save_path=str(save_path))
    except Exception as exc:
      _log.warning("Dashboard save failed at step %d: %s", self.step, exc)

  def _resample_alp_waypoints(self, pred_xyz) -> Optional[np.ndarray]:
    if pred_xyz is None:
      return None
    try:
      arr = np.asarray(pred_xyz, dtype=np.float32)
    except Exception:
      return None
    if arr.ndim != 2 or arr.shape[0] < 10 or arr.shape[1] < 2:
      return None
    indices = np.linspace(0, arr.shape[0] - 1, 10, dtype=int)
    return arr[indices, :2]

  def _build_dashboard_vlm_json(self, hint: dict, risk) -> dict:
    meta_action = str(hint.get("meta_action") or "proceed").lower()
    hazard_type = str(hint.get("hazard_type") or "").lower()
    confidence = float(hint.get("confidence") or 0.0)

    red_light = "red_light" in hazard_type or "traffic_light" in hazard_type or "red light" in hazard_type
    pedestrian = "pedestrian" in hazard_type or "cyclist" in hazard_type
    collision_risk = (hazard_type in {"vehicle", "collision"} or
                      (meta_action in {"stop", "yield"} and not red_light and not pedestrian))
    lead_vehicle = "lead" in hazard_type
    cut_in = "cut_in" in hazard_type or "cut in" in hazard_type
    emergency = "emergency" in hazard_type

    meta_to_action = {
        "stop": "stop",
        "yield": "stop",
        "slow_down": "slow",
        "cautious_proceed": "slow",
        "proceed": "proceed",
    }
    recommended = meta_to_action.get(meta_action, "proceed" if not hint else "slow")

    return {
        "danger_score": float(risk.score),
        "red_light": red_light,
        "red_light_visible": red_light,
        "red_light_applies_to_ego": red_light,
        "pedestrian": pedestrian,
        "pedestrian_in_path": pedestrian,
        "collision_risk": collision_risk,
        "lead_vehicle": lead_vehicle,
        "cut_in_vehicle": cut_in,
        "emergency_vehicle": emergency,
        "recommended_action": recommended,
        "signal_confidence": confidence if red_light else 0.0,
        "reasoning": str(hint.get("reasoning") or "")[:120],
    }

  def _dashboard_mode(self, final_control) -> str:
    meta = getattr(final_control, "meta", {})
    if meta.get("semantic_arbiter"):
      reason = meta.get("override_reason", "intervention")
      return f"Alpamayo {reason}"
    return "TF++ active"

  def _dashboard_final_action(self, final_control, base_control) -> str:
    brake_override = final_control.brake - base_control.brake > 0.05
    throttle_cut = base_control.throttle - final_control.throttle > 0.05
    if final_control.brake >= 0.9:
      return "stop"
    if brake_override or throttle_cut:
      return "slow"
    return "proceed"
