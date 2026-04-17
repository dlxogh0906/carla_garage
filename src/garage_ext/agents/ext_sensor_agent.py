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
from typing import Any

from sensor_agent import SensorAgent  # upstream, in team_code/

from ..config.ext_config import ExtConfig, load_experiment_config
from ..modules import base as mb  # Observation/Plan/Control dataclasses
from ..modules import vlm, risk, safety  # noqa: F401  # force registrations
from ..pipeline import ExtPipeline


def get_entry_point():
    return "ExtSensorAgent"


class ExtSensorAgent(SensorAgent):
    """Upstream SensorAgent + team modules (VLM / risk / safety)."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        self._ext_cfg = self._load_ext_config()
        self._ext_pipeline = ExtPipeline(self._ext_cfg) if self._ext_cfg else None

    @staticmethod
    def _load_ext_config() -> ExtConfig | None:
        cfg_path = os.environ.get("GARAGE_EXT_CONFIG")
        if not cfg_path:
            return None
        return load_experiment_config(cfg_path)

    def run_step(self, input_data, timestamp, sensors=None, plan=None):
        control = super().run_step(input_data, timestamp, sensors=sensors, plan=plan)
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
