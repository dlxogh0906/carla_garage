import time

from garage_ext.modules.base import Control, Observation, RiskReport
from garage_ext.modules.safety.semantic_arbiter import SemanticArbiter


def _obs_with_hint(hint):
  return Observation(data={"vlm_hint": hint, "vlm_hint_ts": time.monotonic()})


def test_yield_lateral_clearance_does_not_apply_longitudinal_override():
  arbiter = SemanticArbiter(risk_threshold=0.45)
  base = Control(throttle=1.0, brake=0.0, steer=0.12)
  risk = RiskReport(score=0.9)
  obs = _obs_with_hint({
      "meta_action": "yield",
      "confidence": 0.75,
      "hazard_type": "none",
      "target_speed_cap_mps": None,
      "reasoning": "Nudge to the left to clear the stopped van blocking our lane",
      "traj_analysis": {"fwd_10": 0.3},
  })

  out = arbiter.filter(base, risk, obs)

  assert out.throttle == base.throttle
  assert out.brake == base.brake
  assert out.steer == base.steer
  assert out.meta["override_reason"] == "yield_lateral_clearance"


def test_yield_crossing_pedestrian_still_brakes():
  arbiter = SemanticArbiter(risk_threshold=0.45)
  base = Control(throttle=1.0, brake=0.0, steer=0.12)
  risk = RiskReport(score=0.9)
  obs = _obs_with_hint({
      "meta_action": "yield",
      "confidence": 0.75,
      "hazard_type": "pedestrian",
      "target_speed_cap_mps": None,
      "reasoning": "Yield for the pedestrian crossing at the crosswalk",
      "traj_analysis": {"fwd_10": 0.3},
  })

  out = arbiter.filter(base, risk, obs)

  assert out.throttle == 0.0
  assert out.brake == 0.5
  assert out.steer == base.steer
  assert out.meta["override_reason"] == "yield"


def test_cautious_proceed_without_speed_cap_is_pass_through():
  arbiter = SemanticArbiter(risk_threshold=0.45)
  base = Control(throttle=1.0, brake=0.0, steer=-0.05)
  risk = RiskReport(score=0.9)
  obs = _obs_with_hint({
      "meta_action": "cautious_proceed",
      "confidence": 0.55,
      "hazard_type": "none",
      "target_speed_cap_mps": None,
      "reasoning": "Keep distance to the cyclist since it is directly ahead in our lane",
      "traj_analysis": {"fwd_10": 0.2},
  })

  out = arbiter.filter(base, risk, obs)

  assert out.throttle == base.throttle
  assert out.brake == base.brake
  assert out.steer == base.steer
  assert out.meta["override_reason"] == "cautious_no_speed_cap"
