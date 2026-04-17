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
import os
import pathlib
import re
from typing import Any

import cv2
import numpy as np

from sensor_agent import SensorAgent  # upstream, in team_code/

from garage_ext.config.ext_config import ExtConfig, load_experiment_config
from garage_ext.modules import base as mb
from garage_ext.modules import vlm, risk, safety, image_enhancer  # noqa: F401
from garage_ext.pipeline import ExtPipeline
from garage_ext.registry import build as registry_build


def get_entry_point():
  return "ExtSensorAgent"


class ExtSensorAgent(SensorAgent):
  """Upstream SensorAgent + team modules (image enhancer / VLM / risk / safety)."""

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
    self._ext_cfg = self._load_ext_config()
    self._ext_pipeline = ExtPipeline(self._ext_cfg) if self._ext_cfg else None
    self._image_enhancer = (registry_build("image_enhancer", self._ext_cfg.image_enhancer, **
                                           self._ext_cfg.image_enhancer_kwargs)
                            if self._ext_cfg and self._ext_cfg.image_enhancer else None)
    self._save_interval = 4
    self._compare_counter = 0
    self._compare_root = (pathlib.Path(os.environ.get("SAVE_PATH", "")) if os.environ.get("SAVE_PATH") else None)
    self._route_tag = self._extract_route_tag(path_to_conf_file)
    self._compare_save_path = self._build_compare_save_path()

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
    panels = []
    for key, bgr_orig, bgr_enh in frames:
      label = key.replace("rgb_", "").upper()
      h = bgr_orig.shape[0]

      # 각 프레임에 라벨 텍스트
      orig_labeled = bgr_orig.copy()
      enh_labeled = bgr_enh.copy()
      cv2.putText(orig_labeled, f"ORIGINAL [{label}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
      cv2.putText(enh_labeled, f"ENHANCED [{label}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

      # 구분선 (흰색 세로줄 4px)
      divider = np.full((h, 4, 3), 255, dtype=np.uint8)
      panels.append(np.concatenate([orig_labeled, divider, enh_labeled], axis=1))

    self._compare_save_path.mkdir(parents=True, exist_ok=True)
    comparison = np.concatenate(panels, axis=0)
    cv2.imwrite(str(self._compare_save_path / f"{self._compare_counter:05d}.png"), comparison)

  def run_step(self, input_data, timestamp, sensors=None, plan=None):
    # ★ 이미지 보정: upstream 모델이 보기 전에 적용
    input_data = self._apply_image_enhancement(input_data)

    control = super().run_step(input_data, timestamp, sensors=sensors)
    if self._ext_pipeline is None:
      return control

    # Wrap upstream state into our neutral dataclasses. We deliberately
    # keep this thin: modules that need more can read self (or upstream
    # attrs) via the `obs.data` bag.
    obs = mb.Observation(data={"input_data": input_data, "timestamp": timestamp, "agent": self})
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
    return control
