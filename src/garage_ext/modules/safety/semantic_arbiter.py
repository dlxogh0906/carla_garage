"""SemanticArbiter — freshness- and consistency-gated safety module for Phase 3.

Design principles (dual-system VLA):
  - Emergency TTC braking is TF++'s job. This module does NOT replicate it.
  - Alpamayo provides preemptive semantic intent: slow, yield, stop.
  - Intervention is gated by staleness, confidence, and reasoning/trajectory
    consistency to prevent low-quality hints from overriding TF++.
  - Steer is never modified. Only throttle/brake actuation.

Gate order (any fail → pass-through):
  1. hint present and non-empty
  2. hint_age < stale_s
  3. hint confidence >= min_confidence
  4. consistency: meta_action and trajectory do not contradict each other
  5. risk.score >= risk_threshold

Action mapping (meta_action):
  stop             → brake floor 1.0, throttle 0
  yield            → brake floor 0.5, throttle 0
                    except lateral-clearance hints that need TF++ steering
  slow_down        → throttle capped, light brake if speed over cap
  cautious_proceed → pass-through unless an explicit speed cap is provided
  proceed / None   → pass-through

Config kwargs (via YAML ext_config safety_kwargs):
  risk_threshold  : float, default 0.45
  stale_s         : float, default 8.0
  min_confidence  : float, default 0.4
  stop_brake      : float, default 1.0
  yield_brake     : float, default 0.5
  slow_brake      : float, default 0.2   (applied only if over speed cap)
  caution_throttle_scale : float, default 0.5
"""

import logging
import time
from typing import Any, Optional

from ...registry import register
from ..base import Control, Observation, RiskReport

log = logging.getLogger(__name__)

_LATERAL_CLEARANCE_TERMS = (
    "nudge",
    "clear",
    "go around",
    "go round",
    "pass the",
    "drive around",
    "move left",
    "move right",
    "left to clear",
    "right to clear",
    "blocking",
    "parked",
    "stopped vehicle",
    "stopped car",
    "stopped van",
    "stopped bus",
    "stopped truck",
)

_HARD_YIELD_TERMS = (
    "crossing",
    "crosswalk",
    "pedestrian",
    "oncoming",
    "red light",
    "stop sign",
    "intersection",
    "merge",
    "cut-in",
    "cut in",
    "entering our lane",
    "gap",
)


@register("safety", "semantic_arbiter")
class SemanticArbiter:
    """Semantic safety arbiter gated by freshness and consistency."""

    def __init__(
        self,
        risk_threshold: float = 0.45,
        stale_s: float = 8.0,
        min_confidence: float = 0.4,
        stop_brake: float = 1.0,
        yield_brake: float = 0.5,
        slow_brake: float = 0.2,
        caution_throttle_scale: float = 0.5,
        **_: Any,
    ) -> None:
        self._risk_threshold = risk_threshold
        self._stale_s = stale_s
        self._min_confidence = min_confidence
        self._stop_brake = stop_brake
        self._yield_brake = yield_brake
        self._slow_brake = slow_brake
        self._caution_throttle_scale = caution_throttle_scale

    def filter(self, control: Control, risk: RiskReport, obs: Observation) -> Control:
        hint = obs.data.get("vlm_hint", {})
        hint_ts = obs.data.get("vlm_hint_ts", 0.0)
        hint_age = time.monotonic() - hint_ts if hint_ts else float("inf")

        log_ctx = {
            "hint_age": round(hint_age, 2),
            "confidence": hint.get("confidence", None),
            "meta_action": hint.get("meta_action", None),
            "target_speed_cap": hint.get("target_speed_cap_mps", None),
            "risk_score": round(risk.score, 3),
            "override_applied": False,
            "override_reason": "none",
            "consistency_pass": None,
        }

        # Gate 1: hint present
        if not hint:
            return self._passthrough(control, log_ctx, "no_hint")

        # Gate 2: staleness
        if hint_age > self._stale_s:
            return self._passthrough(control, log_ctx, "stale_hint")

        # Gate 3: confidence
        conf = float(hint.get("confidence", 1.0))
        if conf < self._min_confidence:
            return self._passthrough(control, log_ctx, "low_confidence")

        # Gate 4: reasoning/trajectory consistency
        consistent = self._consistency_check(hint)
        log_ctx["consistency_pass"] = consistent
        if not consistent:
            return self._passthrough(control, log_ctx, "inconsistent")

        # Gate 5: risk threshold
        if risk.score < self._risk_threshold:
            return self._passthrough(control, log_ctx, "below_threshold")

        meta = hint.get("meta_action", "proceed")
        speed_cap: Optional[float] = hint.get("target_speed_cap_mps")
        current_speed = self._get_speed(obs)

        if meta == "stop":
            out = Control(
                steer=control.steer,
                throttle=0.0,
                brake=max(control.brake, self._stop_brake),
                meta={**control.meta, "semantic_arbiter": True, "override_reason": "stop"},
            )
            log_ctx.update(override_applied=True, override_reason="stop")

        elif meta == "yield":
            if self._is_lateral_clearance_yield(hint):
                return self._passthrough(control, log_ctx, "yield_lateral_clearance")
            out = Control(
                steer=control.steer,
                throttle=0.0,
                brake=max(control.brake, self._yield_brake),
                meta={**control.meta, "semantic_arbiter": True, "override_reason": "yield"},
            )
            log_ctx.update(override_applied=True, override_reason="yield")

        elif meta == "slow_down":
            new_throttle = control.throttle
            new_brake = control.brake
            if speed_cap is not None and current_speed is not None and current_speed > speed_cap:
                new_throttle = 0.0
                new_brake = max(control.brake, self._slow_brake)
                log_ctx["override_reason"] = f"slow_down(over_cap={speed_cap:.1f})"
            else:
                new_throttle = min(control.throttle, 0.5)
                log_ctx["override_reason"] = "slow_down(throttle_cap)"
            out = Control(
                steer=control.steer,
                throttle=new_throttle,
                brake=new_brake,
                meta={**control.meta, "semantic_arbiter": True, "override_reason": log_ctx["override_reason"]},
            )
            log_ctx["override_applied"] = True

        elif meta == "cautious_proceed":
            if speed_cap is None:
                return self._passthrough(control, log_ctx, "cautious_no_speed_cap")
            new_throttle = control.throttle * self._caution_throttle_scale
            out = Control(
                steer=control.steer,
                throttle=new_throttle,
                brake=control.brake,
                meta={**control.meta, "semantic_arbiter": True, "override_reason": "cautious_proceed"},
            )
            log_ctx.update(override_applied=True, override_reason="cautious_proceed")

        else:
            return self._passthrough(control, log_ctx, f"meta={meta}_no_action")

        log.info(
            "SemanticArbiter OVERRIDE | %s | score=%.2f conf=%.2f age=%.1fs "
            "throttle %.2f→%.2f brake %.2f→%.2f | %s",
            log_ctx["override_reason"],
            risk.score, conf, hint_age,
            control.throttle, out.throttle,
            control.brake, out.brake,
            hint.get("reasoning", "")[:80],
        )
        return out

    # ------------------------------------------------------------------

    def _consistency_check(self, hint: dict) -> bool:
        """Return False if meta_action and trajectory intent clearly contradict.

        Strong intervention (stop/yield) is flagged as inconsistent if the
        trajectory shows sustained forward motion (fwd_10 > 3 m).
        Other combinations are accepted — the model may be anticipating future events.
        """
        meta = hint.get("meta_action", "proceed")
        ta = hint.get("traj_analysis", {})
        if not ta:
            return True  # no trajectory data → can't check, trust meta_action

        fwd = ta.get("fwd_10", 999.0)

        if meta in ("stop", "yield") and fwd > 3.0:
            log.debug("Consistency fail: meta=%s but fwd_10=%.2f m", meta, fwd)
            return False

        return True

    @staticmethod
    def _is_lateral_clearance_yield(hint: dict) -> bool:
        """Detect yield hints that actually ask TF++ to steer around a blockage."""
        if hint.get("target_speed_cap_mps") is not None:
            return False

        hazard = str(hint.get("hazard_type") or "none").lower()
        text_parts = [
            str(hint.get("reasoning") or ""),
            " ".join(str(x) for x in hint.get("cot") or []),
        ]
        text = " ".join(text_parts).lower()

        has_lateral_clearance = any(term in text for term in _LATERAL_CLEARANCE_TERMS)
        has_hard_yield = any(term in text for term in _HARD_YIELD_TERMS)
        static_or_unknown_hazard = hazard in ("none", "static", "obstacle")

        return has_lateral_clearance and static_or_unknown_hazard and not has_hard_yield

    def _get_speed(self, obs: Observation) -> Optional[float]:
        """Try to read current ego speed (m/s) from speedometer sensor."""
        input_data = obs.data.get("input_data", {})
        spd = input_data.get("carla_speedometer")
        if spd is not None:
            try:
                _, arr = spd
                return float(arr[0]) if hasattr(arr, "__len__") else float(arr)
            except Exception:
                pass
        agent = obs.data.get("agent")
        if agent is not None:
            try:
                log = getattr(agent, "state_log", None)
                if log and len(log) > 0:
                    # state_log stores (x, y, theta, speed) or similar
                    entry = log[-1]
                    if len(entry) >= 4:
                        return float(entry[3])
            except Exception:
                pass
        return None

    @staticmethod
    def _passthrough(control: Control, log_ctx: dict, reason: str) -> Control:
        log_ctx["override_reason"] = reason
        log.debug("SemanticArbiter pass-through: %s", reason)
        return Control(
            steer=control.steer,
            throttle=control.throttle,
            brake=control.brake,
            meta={**control.meta, "override_reason": reason},
        )
