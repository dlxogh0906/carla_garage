"""Protocol (duck-typed interface) definitions for pluggable modules.

Using Protocol instead of ABC keeps teammates free to ship a class that
simply has the right methods, without forced inheritance. The pipeline
only ever relies on these signatures.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class Observation:
  """Whatever the agent already computed per-step: images, LiDAR, BEV, speed, etc.

    Kept as an open bag so different modules can read what they need.
    """
  data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
  """Plan produced by the upstream model (waypoints or path+target_speed)."""
  waypoints: Any = None
  target_speed: Any = None
  meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskReport:
  score: float = 0.0
  reasons: List[str] = field(default_factory=list)
  extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Control:
  """Final actuation the agent will return to the simulator."""
  steer: float = 0.0
  throttle: float = 0.0
  brake: float = 0.0
  meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VLMModule(Protocol):
  """Vision-language module. Returns a free-form dict merged into Observation."""

  def infer(self, obs: Observation) -> Dict[str, Any]:
    ...


@runtime_checkable
class RiskEstimator(Protocol):

  def estimate(self, obs: Observation, plan: Plan) -> RiskReport:
    ...


@runtime_checkable
class SafetyFilter(Protocol):
  """Last-mile guard: may modify Control based on the risk report."""

  def filter(self, control: Control, risk: RiskReport, obs: Observation) -> Control:
    ...
