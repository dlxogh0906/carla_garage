"""Team8MetaActionQwenAgent: TF++ + teammate-style 8 meta-action Qwen VLA.

Deliberately small and separate from QwenSensorAgent:
- front camera only
- optional ClassicCV image enhancement for Eunsu-style experiments
- no rear camera / emergency-yield logic
- Qwen returns one digit 0-7
- digit maps to a speed multiplier applied to TF++ target speed
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
from typing import Optional

import numpy as np

from sensor_agent import SensorAgent

from garage_ext.vlm_intervention.qwen_8meta_action_client import META_ACTIONS, Qwen8MetaActionClient
from garage_ext.vlm_intervention.ttc import compute_simplified_ttc, get_front_distance, is_risky_ttc

_log = logging.getLogger(__name__)


def get_entry_point() -> str:
    return "Team8MetaActionQwenAgent"


class Team8MetaActionQwenAgent(SensorAgent):
    """TF++ agent with teammate 8-action Qwen speed intervention."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self._vlm_client = None
        self._log_f = None
        self._qwen_speed_scale = 1.0
        self._qwen_speed_scale_applied = 1.0
        self._captured_tfpp_target_speed = float("nan")
        self._captured_final_target_speed = float("nan")
        self._qwen_result = {
            "action_idx": 0,
            "action_name": "proceed",
            "speed_scale": 1.0,
            "raw_response": "",
            "reason": "not_ready",
        }
        self._image_enhancer = None
        self._image_enhancer_name = ""

        super().setup(
            path_to_conf_file,
            route_index=route_index,
            traffic_manager=traffic_manager,
        )

        self._vlm_enabled = os.environ.get("QWEN_VLM_ENABLED", "1") == "1"
        self._ttc_threshold = float(os.environ.get("QWEN_TTC_THRESHOLD", "3.0"))
        self._meta_every_n = max(1, int(os.environ.get("QWEN_8META_EVERY_N_STEPS", "20")))
        self._lateral_thresh = float(os.environ.get("QWEN_LATERAL_THRESH", "2.0"))
        self._front_max_distance = float(os.environ.get("QWEN_FRONT_MAX_DISTANCE", "80.0"))
        self._dashboard_interval = max(1, int(os.environ.get("EXT_DASHBOARD_INTERVAL", "4")))
        self._setup_image_enhancer()

        self._route_tag = self._extract_route_tag(path_to_conf_file)
        self._route_save_path = self._build_route_save_path()
        self._open_log()

        if self._vlm_enabled:
            model_name = os.environ.get("QWEN_MODEL", "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct")
            device = os.environ.get("QWEN_VLM_DEVICE", "auto")
            thinking = os.environ.get("QWEN_VLM_THINKING", "0") == "1"
            self._vlm_client = Qwen8MetaActionClient(model_name, device=device, enable_thinking=thinking)
            _log.info(
                "Team8MetaActionQwenAgent initialised: model=%s device=%s ttc_thr=%.2f every=%d enhancer=%s",
                model_name,
                device,
                self._ttc_threshold,
                self._meta_every_n,
                self._image_enhancer_name or "off",
            )

        self._patch_speed_scale()

    def destroy(self) -> None:
        if self._log_f is not None:
            self._log_f.close()
            self._log_f = None
        if self._vlm_client is not None:
            self._vlm_client.shutdown()
        super().destroy()

    def sensors(self):
        return super().sensors()

    def run_step(self, input_data, timestamp, sensors=None):
        input_data = self._maybe_enhance_input_data(input_data)
        control = super().run_step(input_data, timestamp, sensors=sensors)

        ego_speed = self._get_ego_speed(input_data)
        bb_list = list(self.bb_buffer[-1]) if len(self.bb_buffer) > 0 else []
        front_distance = get_front_distance(
            bb_list,
            lateral_thresh=self._lateral_thresh,
            max_distance=self._front_max_distance,
        )
        ttc = compute_simplified_ttc(front_distance, ego_speed)
        risky = is_risky_ttc(ttc, self._ttc_threshold)

        vlm_called = False
        if risky and self._vlm_client is not None and self._vlm_client.is_ready:
            front_rgb = self._get_front_image(input_data)
            if front_rgb is not None:
                vlm_called = self._vlm_client.request(
                    front_rgb,
                    step=self.step,
                    min_step_interval=self._meta_every_n,
                )

        if self._vlm_client is not None:
            self._qwen_result = self._vlm_client.get_latest()
        elif not self._vlm_enabled:
            self._qwen_result = {
                "action_idx": 0,
                "action_name": "proceed",
                "speed_scale": 1.0,
                "raw_response": "",
                "reason": "vlm_disabled",
            }

        self._qwen_speed_scale = float(self._qwen_result.get("speed_scale", 1.0)) if risky else 1.0
        tfpp_speed = getattr(self, "_captured_tfpp_target_speed", float("nan"))
        applied_scale = getattr(self, "_qwen_speed_scale_applied", self._qwen_speed_scale)
        final_speed = getattr(self, "_captured_final_target_speed", float("nan"))
        if not np.isfinite(final_speed) and np.isfinite(tfpp_speed):
            final_speed = tfpp_speed * applied_scale

        self._log_step(
            ego_speed=ego_speed,
            front_distance=front_distance,
            ttc=ttc,
            risky=risky,
            vlm_called=vlm_called,
            tfpp_speed=tfpp_speed,
            final_speed=final_speed,
            applied_scale=applied_scale,
            control=control,
        )
        self._save_dashboard(input_data, ttc, risky, ego_speed, front_distance, control)
        return control

    def _patch_speed_scale(self) -> None:
        nets = getattr(self, "nets", None)
        if not nets:
            return
        net = nets[0]
        original = getattr(net, "control_pid_direct", None)
        if original is None or getattr(net, "_qwen_8meta_speed_patch", False):
            return

        agent = self

        def _wrapped(pred_checkpoints, pred_target_speed, speed, *args, **kwargs):
            agent._captured_tfpp_target_speed = float(pred_target_speed)
            scaled = pred_target_speed * agent._qwen_speed_scale
            agent._qwen_speed_scale_applied = float(agent._qwen_speed_scale)
            agent._captured_final_target_speed = float(scaled)
            return original(pred_checkpoints, scaled, speed, *args, **kwargs)

        net.control_pid_direct = _wrapped
        net._qwen_8meta_speed_patch = True

    def _open_log(self) -> None:
        if self._route_save_path is None:
            return
        self._route_save_path.mkdir(parents=True, exist_ok=True)
        self._log_f = open(self._route_save_path / "qwen_intervention.jsonl", "a", buffering=1)

    def _log_step(
        self,
        ego_speed: float,
        front_distance: float,
        ttc: float,
        risky: bool,
        vlm_called: bool,
        tfpp_speed: float,
        final_speed: float,
        applied_scale: float,
        control,
    ) -> None:
        if self._log_f is None:
            return
        result = dict(self._qwen_result or {})
        action_idx = int(result.get("action_idx", 0))
        action_name = str(result.get("action_name", META_ACTIONS[action_idx][0]))
        entry = {
            "experiment": "team8_meta_action_qwen",
            "image_enhancer": self._image_enhancer_name or "off",
            "step": int(self.step),
            "ego_speed": _round_or_none(ego_speed, 3),
            "front_distance": _round_or_none(front_distance, 2, cap=999),
            "ttc": _round_or_none(ttc, 3, cap=999),
            "is_risky": bool(risky),
            "vlm_called": bool(vlm_called),
            "vlm_ready": bool(self._vlm_client is not None and self._vlm_client.is_ready),
            "vlm_load_error": self._vlm_client.load_error if self._vlm_client is not None else None,
            "vlm_trigger": "ttc" if risky else "none",
            "prompt_mode": "team8_meta_action_digit",
            "action_idx": action_idx,
            "action_name": action_name,
            "qwen_intervene": action_idx != 0,
            "qwen_requested_scale": _round_or_none(float(result.get("speed_scale", 1.0)), 4),
            "speed_scale": _round_or_none(applied_scale, 4),
            "tfpp_target_speed": _round_or_none(tfpp_speed, 3),
            "final_target_speed": _round_or_none(final_speed, 3),
            "risk_level": result.get("risk_level", "low"),
            "reason": result.get("reason", ""),
            "qwen_raw_response": str(result.get("raw_response", ""))[:200],
            "qwen_request_step": result.get("request_step"),
            "qwen_request_trigger": result.get("request_trigger"),
            "control_throttle": _round_or_none(getattr(control, "throttle", None), 4),
            "control_brake": _round_or_none(getattr(control, "brake", None), 4),
            "control_steer": _round_or_none(getattr(control, "steer", None), 4),
        }
        if result.get("benchmark") is not None:
            entry["qwen_benchmark"] = result["benchmark"]
        self._log_f.write(json.dumps(entry) + "\n")

    def _setup_image_enhancer(self) -> None:
        name = os.environ.get("QWEN_8META_IMAGE_ENHANCER", "").strip().lower()
        if not name:
            return
        if name not in {"classic_cv", "classiccv"}:
            _log.warning("Unsupported QWEN_8META_IMAGE_ENHANCER=%s; enhancement disabled", name)
            return
        try:
            from image_enhancement_module.classic_cv_enhancer import ClassicCVEnhancer

            self._image_enhancer = ClassicCVEnhancer()
            self._image_enhancer_name = "classic_cv"
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("ClassicCV enhancer load failed: %s", exc)
            self._image_enhancer = None
            self._image_enhancer_name = ""

    def _maybe_enhance_input_data(self, input_data):
        if self._image_enhancer is None or not isinstance(input_data, dict):
            return input_data
        rgb = input_data.get("rgb_front")
        if rgb is None:
            return input_data
        try:
            ts, frame = rgb
            enhanced = np.array(frame, copy=True)
            bgr = enhanced[:, :, :3].astype(np.uint8)
            enhanced[:, :, :3] = self._image_enhancer.enhance(bgr)
            patched = dict(input_data)
            patched["rgb_front"] = (ts, enhanced)
            return patched
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("ClassicCV enhancement failed at step=%s: %s", getattr(self, "step", "?"), exc)
            return input_data

    def _save_dashboard(self, input_data, ttc, risky, ego_speed, front_distance, control) -> None:
        if (
            self._route_save_path is None
            or os.environ.get("QWEN_SAVE_DASHBOARD", "1") != "1"
            or self.step % self._dashboard_interval != 0
        ):
            return
        try:
            from garage_ext.visualization.qwen_dashboard import render_qwen_dashboard

            image_rgb = self._get_front_image(input_data)
            out_dir = self._route_save_path / "dashboard"
            out_path = out_dir / f"{self.step:05d}.png"
            result = self._qwen_result or {}
            render_qwen_dashboard(
                {
                    "history_key": str(self._route_save_path),
                    "step": self.step,
                    "image": image_rgb,
                    "ttc": ttc,
                    "ttc_threshold": self._ttc_threshold,
                    "risky": risky,
                    "ego_speed": ego_speed,
                    "front_distance": front_distance,
                    "speed_scale": self._qwen_speed_scale_applied,
                    "action_id": result.get("action_idx", 0),
                    "action": result.get("action_name", "proceed"),
                    "tfpp_target_speed": getattr(self, "_captured_tfpp_target_speed", float("nan")),
                    "final_target_speed": getattr(self, "_captured_final_target_speed", float("nan")),
                    "control_throttle": getattr(control, "throttle", None),
                    "control_brake": getattr(control, "brake", None),
                    "control_steer": getattr(control, "steer", None),
                },
                save_path=str(out_path),
            )
        except Exception as exc:
            _log.warning("Team8 dashboard save failed at step %s: %s", self.step, exc)

    @staticmethod
    def _get_ego_speed(input_data: dict) -> float:
        for key in ("speed", "speedometer", "carla_speedometer"):
            spd = input_data.get(key)
            if spd is None:
                continue
            try:
                _, payload = spd
                if isinstance(payload, dict):
                    v = float(payload.get("speed", 0.0))
                elif hasattr(payload, "__len__"):
                    v = float(payload[0])
                else:
                    v = float(payload)
                return max(0.0, v)
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _get_front_image(input_data: dict) -> Optional[np.ndarray]:
        rgb = input_data.get("rgb_front")
        if rgb is None:
            return None
        try:
            _, frame = rgb
            import cv2

            bgr = frame[:, :, :3].astype(np.uint8)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    @staticmethod
    def _extract_route_tag(path_to_conf_file: str) -> str:
        text = str(path_to_conf_file or "")
        parts = text.split("+")
        if len(parts) > 1:
            match = re.search(r"(RouteScenario_\d+_rep\d+)", parts[-1])
            if match:
                return match.group(1)
            return parts[-1] or "unknown_route"
        match = re.search(r"(RouteScenario_[^/\\]+?)(?:\.xml|$)", text)
        if match:
            return match.group(1)
        return "unknown_route"

    def _build_route_save_path(self) -> Optional[pathlib.Path]:
        if getattr(self, "save_path", None) is not None:
            return pathlib.Path(self.save_path)
        save_env = os.environ.get("SAVE_PATH", "")
        if not save_env:
            return None
        return pathlib.Path(save_env) / self._route_tag


def _round_or_none(value, digits: int, cap: Optional[float] = None):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    if cap is not None and x >= cap:
        return None
    return round(x, digits)
