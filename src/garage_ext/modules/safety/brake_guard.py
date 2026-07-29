"""BrakeGuardSafety — last-mile brake intervention for Phase 2.

When risk.score exceeds `risk_threshold` AND the VLM hint is fresh enough,
clamps `control.brake` to at least `brake_floor` and reduces throttle
proportionally.

Config kwargs (via YAML ext_config safety_kwargs):
  risk_threshold : float, default 0.5   — minimum risk.score to intervene
  brake_floor    : float, default 0.4   — minimum brake value applied
  hint_stale_s   : float, default 8.0   — ignore hints older than this (seconds)
  throttle_scale : float, default 0.0   — throttle multiplier when braking (0 = cut fully)
"""

import logging
import time
from typing import Any

from ...registry import register
from ..base import Control, Observation, RiskReport

log = logging.getLogger(__name__)


@register("safety", "brake_guard")
class BrakeGuardSafety:
    """Applies a minimum brake value when VLM risk score is high."""

    def __init__(
        self,
        risk_threshold: float = 0.5,
        brake_floor: float = 0.4,
        hint_stale_s: float = 8.0,
        throttle_scale: float = 0.0,
        **_: Any,
    ) -> None:
        self._risk_threshold = risk_threshold
        self._brake_floor = brake_floor
        self._hint_stale_s = hint_stale_s
        self._throttle_scale = throttle_scale

    def filter(self, control: Control, risk: RiskReport, obs: Observation) -> Control:
        if risk.score < self._risk_threshold:
            return control

        # Check hint freshness via vlm_hint_ts stored by AlpamayoZmqVLM
        hint_ts = obs.data.get("vlm_hint_ts", 0.0)
        if hint_ts and (time.monotonic() - hint_ts) > self._hint_stale_s:
            log.debug("BrakeGuard: risk=%.2f but hint is stale — skipping", risk.score)
            return control

        new_brake = max(control.brake, self._brake_floor)
        new_throttle = control.throttle * self._throttle_scale

        log.info(
            "BrakeGuard INTERVENE score=%.2f brake %.2f→%.2f throttle %.2f→%.2f | %s",
            risk.score,
            control.brake, new_brake,
            control.throttle, new_throttle,
            "; ".join(risk.reasons[:3]),
        )

        return Control(
            steer=control.steer,
            throttle=new_throttle,
            brake=new_brake,
            meta={**control.meta, "brake_guard": True, "risk_score": risk.score},
        )
