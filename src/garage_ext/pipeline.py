"""Compose VLM + risk + safety modules around upstream perception/planning.

Flow per step:
    upstream_plan  ->  pipeline.run(obs, plan, base_control)  ->  final_control

The pipeline intentionally does NOT own perception or planning. Those
stay in upstream `team_code/`. We only inject extra signals (VLM),
score them (risk), and guard the output (safety).
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
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
    self._step = 0
    self._ilog_file = None
    self._ilog_path: Optional[Path] = None

  def run(self, obs: Observation, plan: Plan, base_control: Control) -> PipelineOutputs:
    self._step += 1

    # TTC must be computed before VLM so the VLM trigger can read obs.data["ttc"]
    self._precompute_ttc(obs)

    vlm_info: dict = {}
    if self.vlm is not None:
      vlm_info = self.vlm.infer(obs)
      obs.data.update(vlm_info)   # vlm_hint, vlm_hint_ts, vlm_step → top-level keys
    risk = self.risk.estimate(obs, plan)
    control = self.safety.filter(base_control, risk, obs)

    self._log_intervention(obs, base_control, control, risk)

    return PipelineOutputs(control=control, risk=risk, vlm_info=vlm_info)

  # ------------------------------------------------------------------
  # Intervention logging
  # ------------------------------------------------------------------

  def _log_intervention(
      self,
      obs: Observation,
      base: Control,
      out: Control,
      risk: RiskReport,
  ) -> None:
    """Write one JSONL line per step that has an active VLM hint."""
    hint = obs.data.get("vlm_hint", {})
    if not hint and risk.score == 0.0:
      return  # nothing to log

    hint_ts = obs.data.get("vlm_hint_ts", 0.0)
    hint_age = round(time.monotonic() - hint_ts, 2) if hint_ts else None
    override = self._detect_override(base, out)

    entry = {
      "step":          self._step,
      "timestamp":     obs.data.get("timestamp", 0.0),
      "route_id":      self._route_id(obs),
      # VLM signal
      "meta_action":   hint.get("meta_action"),
      "hazard_type":   hint.get("hazard_type"),
      "confidence":    hint.get("confidence"),
      "expiry_sec":    hint.get("expiry_sec"),
      "target_speed_cap_mps": hint.get("target_speed_cap_mps"),
      "reasoning":     hint.get("reasoning", "")[:120],
      "traj_stop":     (hint.get("traj_analysis") or {}).get("stop_intent"),
      "traj_slow":     (hint.get("traj_analysis") or {}).get("slow_intent"),
      "traj_fwd10":    (hint.get("traj_analysis") or {}).get("fwd_10"),
      "hint_age_s":    hint_age,
      # Trajectory disagreement
      "dis_score":      round(risk.extra.get("disagreement", {}).get("score", 0.0), 4),
      "dis_prog1s":     risk.extra.get("disagreement", {}).get("prog_gap_1s"),
      "dis_prog2s":     risk.extra.get("disagreement", {}).get("prog_gap_2s"),
      "dis_lat1s":      risk.extra.get("disagreement", {}).get("lat_gap_1s"),
      "dis_available":  risk.extra.get("disagreement", {}).get("available", False),
      "dis_reason":     risk.extra.get("disagreement", {}).get("reason"),
      "tf_pred_path_kind": obs.data.get("tf_pred_path_kind"),
      "tf_pred_path_avail": obs.data.get("tf_pred_path") is not None,
      "tf_pred_wp_avail": obs.data.get("tf_pred_wp") is not None,
      "tf_pred_checkpoint_avail": obs.data.get("tf_pred_checkpoints") is not None,
      "tf_pred_target_speed_mps": obs.data.get("tf_pred_target_speed_mps"),
      # Risk
      "risk_score":    round(risk.score, 4),
      "risk_reasons":  risk.reasons[:3],
      # Control delta
      "tf_throttle":   round(base.throttle, 3),
      "tf_brake":      round(base.brake, 3),
      "tf_steer":      round(base.steer, 3),
      "out_throttle":  round(out.throttle, 3),
      "out_brake":     round(out.brake, 3),
      "out_steer":     round(out.steer, 3),
      "override":      override,
      "arbiter_reason": out.meta.get("override_reason") or (out.meta.get("semantic_arbiter") and "intervened"),
    }

    f = self._get_ilog()
    if f is not None:
      f.write(json.dumps(entry) + "\n")
      f.flush()

  @staticmethod
  def _detect_override(base: Control, out: Control) -> bool:
    return (
        abs(out.throttle - base.throttle) > 0.01
        or abs(out.brake - base.brake) > 0.01
    )

  @staticmethod
  def _route_id(obs: Observation) -> Optional[str]:
    agent = obs.data.get("agent")
    return getattr(agent, "_route_tag", None) if agent else None

  def _get_ilog(self):
    if self._ilog_file is not None:
      return self._ilog_file
    save_path = os.environ.get("SAVE_PATH", "")
    if not save_path:
      return None
    p = Path(save_path)
    p.mkdir(parents=True, exist_ok=True)
    self._ilog_path = p / "intervention_log.jsonl"
    self._ilog_file = open(self._ilog_path, "a")  # noqa: SIM115
    return self._ilog_file
