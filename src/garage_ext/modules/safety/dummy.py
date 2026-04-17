"""Placeholder safety filter (pass-through).

A real filter might clamp throttle, force a brake when risk.score exceeds
a threshold, or blend with a rule-based fallback.
"""
from typing import Any

from ...registry import register
from ..base import Control, Observation, RiskReport


@register("safety", "noop")
class NoopSafety:

  def __init__(self, **_: Any) -> None:
    pass

  def filter(self, control: Control, risk: RiskReport, obs: Observation) -> Control:
    return control
