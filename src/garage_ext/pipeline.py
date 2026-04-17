"""Compose VLM + risk + safety modules around upstream perception/planning.

Flow per step:
    upstream_plan  ->  pipeline.run(obs, plan, base_control)  ->  final_control

The pipeline intentionally does NOT own perception or planning. Those
stay in upstream `team_code/`. We only inject extra signals (VLM),
score them (risk), and guard the output (safety).
"""
from dataclasses import dataclass
from typing import Optional

from .config.ext_config import ExtConfig
from .modules.base import (Control, Observation, Plan, RiskEstimator, RiskReport, SafetyFilter, VLMModule)
from .registry import build


@dataclass
class PipelineOutputs:
  control: Control
  risk: RiskReport
  vlm_info: dict


class ExtPipeline:
  """Wires registered modules together according to ExtConfig."""

  def __init__(self, cfg: ExtConfig):
    self.cfg = cfg
    self.vlm: Optional[VLMModule] = (build("vlm", cfg.vlm, **cfg.vlm_kwargs) if cfg.vlm else None)
    self.risk: RiskEstimator = build("risk", cfg.risk or "noop", **cfg.risk_kwargs)
    self.safety: SafetyFilter = build("safety", cfg.safety or "noop", **cfg.safety_kwargs)

  def run(self, obs: Observation, plan: Plan, base_control: Control) -> PipelineOutputs:
    vlm_info: dict = {}
    if self.vlm is not None:
      vlm_info = self.vlm.infer(obs)
      obs.data["vlm"] = vlm_info
    risk = self.risk.estimate(obs, plan)
    control = self.safety.filter(base_control, risk, obs)
    return PipelineOutputs(control=control, risk=risk, vlm_info=vlm_info)
