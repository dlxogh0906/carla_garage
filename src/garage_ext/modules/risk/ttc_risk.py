"""TtcRisk — Time-To-Collision based risk estimator.

TTC is computed from:
  - tf_pred_checkpoints (10, 2): ego future path in ego frame
  - agent.bb_buffer[-1]: detected bounding boxes [x, y, w, h, yaw, vel, brake, cls, score]
  - ego speed from carla_speedometer

TTC → risk score mapping (configurable):
  TTC < ttc_critical  → score = 1.0
  TTC < ttc_high      → score linearly interpolated 0.75..1.0
  TTC < ttc_medium    → score linearly interpolated 0.45..0.75
  TTC >= ttc_medium   → score = 0.0

The vlm_hint meta_action is NOT read here — this module only sets the numeric
score. SemanticArbiter reads meta_action separately to decide HOW to respond.

obs.data keys written:
  "ttc"          : float, minimum TTC across all detected actors (seconds)
  "ttc_actor_bb" : the bounding box that gave the minimum TTC (or None)

Config kwargs (via YAML ext_config risk_kwargs):
  ttc_critical   : float, default 1.5  (s)
  ttc_high       : float, default 3.0  (s)
  ttc_medium     : float, default 5.0  (s)
  collision_r    : float, default 2.5  (m) — ego + actor collision margin
  score_critical : float, default 1.0
  score_high     : float, default 0.75
  score_medium   : float, default 0.45
"""

import logging
from typing import Any, Optional

import numpy as np

from ...registry import register
from ..base import Observation, Plan, RiskReport

log = logging.getLogger(__name__)


@register("risk", "ttc_risk")
class TtcRisk:
    """TTC-based risk estimator. Registered as kind='risk', name='ttc_risk'."""

    def __init__(
        self,
        ttc_critical: float = 1.5,
        ttc_high: float = 3.0,
        ttc_medium: float = 5.0,
        collision_r: float = 2.5,
        score_critical: float = 1.0,
        score_high: float = 0.75,
        score_medium: float = 0.45,
        **_: Any,
    ) -> None:
        self._ttc_critical = ttc_critical
        self._ttc_high = ttc_high
        self._ttc_medium = ttc_medium
        self._collision_r = collision_r
        self._score_critical = score_critical
        self._score_high = score_high
        self._score_medium = score_medium

    def estimate(self, obs: Observation, plan: Plan) -> RiskReport:
        v_ego = self._get_ego_speed(obs)
        bb_list = self._get_bb_list(obs)
        pred_checkpoints = obs.data.get("tf_pred_checkpoints")

        ttc, ttc_bb = self._compute_ttc(pred_checkpoints, bb_list, v_ego)

        # write back for downstream (VLM trigger, dashboard, logging)
        obs.data["ttc"] = ttc
        obs.data["ttc_actor_bb"] = ttc_bb

        score = self._ttc_to_score(ttc)
        reasons = [f"ttc={ttc:.2f}s"] if ttc < self._ttc_medium else []

        log.debug("TtcRisk: ttc=%.2fs score=%.3f v_ego=%.2f bbs=%d",
                  ttc, score, v_ego, len(bb_list) if bb_list else 0)

        return RiskReport(
            score=score,
            reasons=reasons,
            extra={"ttc": ttc, "v_ego": v_ego},
        )

    # ------------------------------------------------------------------

    def _ttc_to_score(self, ttc: float) -> float:
        if ttc <= self._ttc_critical:
            return self._score_critical
        if ttc <= self._ttc_high:
            t = (ttc - self._ttc_critical) / (self._ttc_high - self._ttc_critical)
            return self._score_high + (1.0 - t) * (self._score_critical - self._score_high)
        if ttc <= self._ttc_medium:
            t = (ttc - self._ttc_high) / (self._ttc_medium - self._ttc_high)
            return self._score_medium + (1.0 - t) * (self._score_high - self._score_medium)
        return 0.0

    def _compute_ttc(self, pred_checkpoints, bb_list, v_ego):
        """Return (min_ttc, closest_bb) across all detected actors."""
        if not bb_list:
            return float("inf"), None
        if v_ego < 0.5:
            # ego nearly stopped — no imminent collision from speed
            return float("inf"), None

        wps = None
        if pred_checkpoints is not None:
            try:
                wps = np.asarray(pred_checkpoints, dtype=np.float32)  # (N, 2)
                if wps.ndim != 2 or wps.shape[1] < 2:
                    wps = None
            except Exception:
                wps = None

        min_ttc = float("inf")
        min_bb = None

        for bb in bb_list:
            try:
                bx, by = float(bb[0]), float(bb[1])
                bw, bh = float(bb[2]), float(bb[3])
            except Exception:
                continue

            # Only consider actors ahead of ego (positive x = forward in ego frame)
            if bx < 0.5:
                continue

            r = max(bw, bh) / 2.0 + self._collision_r

            ttc = self._ttc_for_actor(wps, bx, by, r, v_ego)
            if ttc < min_ttc:
                min_ttc = ttc
                min_bb = bb

        return min_ttc, min_bb

    def _ttc_for_actor(self, wps, bx, by, r, v_ego) -> float:
        """Waypoint-path TTC for a single actor, falling back to dist/v_ego."""
        if wps is not None and len(wps) > 0:
            prev = np.zeros(2, dtype=np.float32)
            path_len = 0.0
            for wp in wps:
                seg_len = float(np.linalg.norm(wp - prev))
                path_len += seg_len
                dist_to_actor = float(np.sqrt((wp[0] - bx) ** 2 + (wp[1] - by) ** 2))
                if dist_to_actor < r:
                    return path_len / v_ego
                prev = wp

        # Fallback: straight-line distance from ego origin to actor centroid
        direct_dist = float(np.sqrt(bx ** 2 + by ** 2))
        return direct_dist / v_ego

    # ------------------------------------------------------------------

    @staticmethod
    def _get_ego_speed(obs: Observation) -> float:
        input_data = obs.data.get("input_data", {})
        spd = input_data.get("carla_speedometer")
        if spd is not None:
            try:
                _, arr = spd
                return max(0.0, float(arr[0]) if hasattr(arr, "__len__") else float(arr))
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _get_bb_list(obs: Observation):
        agent = obs.data.get("agent")
        if agent is None:
            return []
        buf = getattr(agent, "bb_buffer", None)
        if not buf or len(buf) == 0:
            return []
        bbs = buf[-1]
        return bbs if bbs is not None else []
