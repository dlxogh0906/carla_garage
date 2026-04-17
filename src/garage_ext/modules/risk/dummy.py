"""Placeholder risk estimator (always returns 0)."""
from typing import Any

from ...registry import register
from ..base import Observation, Plan, RiskReport


@register("risk", "noop")
class NoopRisk:

  def __init__(self, **_: Any) -> None:
    pass

  def estimate(self, obs: Observation, plan: Plan) -> RiskReport:
    return RiskReport(score=0.0, reasons=[])
