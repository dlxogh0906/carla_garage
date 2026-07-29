"""CoT-text risk estimator for Phase 2.

Parses the VLM chain-of-thought string stored in obs.data["vlm_hint"]["cot"]
and returns a RiskReport with score in [0, 1].

Keyword sets are derived from Phase 1 shadow-log analysis:
  - CRITICAL keywords → score 1.0 immediately
  - MODERATE keywords → score accumulates (capped at 0.9)

Config kwargs (via YAML ext_config risk_kwargs):
  critical_threshold : float, default 1.0  (score returned on critical hit)
  moderate_weight    : float, default 0.45 (score per moderate keyword match)
  min_score_for_log  : float, default 0.3  (log.debug below this is suppressed)
"""

import logging
import re
from typing import Any, List

from ...registry import register
from ..base import Observation, Plan, RiskReport

log = logging.getLogger(__name__)

# --- keyword tables (lower-cased, matched as substrings) -------------------

CRITICAL_PHRASES: List[str] = [
    "out of control",
    "vehicle is out of control",
    "hazard",
    "emergency",
    "collision",
    "crash",
    "accident",
]

MODERATE_PHRASES: List[str] = [
    "blocking our lane",
    "blocking the lane",
    "blocked",
    "stopped in our lane",
    "stopped in the lane",
    "stopped vehicle",
    "stationary car",
    "stationary vehicle",
    "stationary object",
    "caution",
    "slow down",
    "unusual situation",
    "construction",
    "obstacle",
    "pedestrian crossing",
    "jaywalking",
    "sudden braking",
    "brake",
    "yielding",
]

# Affirmative opening followed by danger word — catches "Yes, there is a hazard"
_AFFIRM_RE = re.compile(
    r"\b(yes|indeed|definitely|certainly)\b.{0,60}"
    r"(hazard|blocked|stopped|caution|danger|obstacle|out of control)",
    re.IGNORECASE,
)


def _score_cot(cot: str) -> tuple[float, List[str]]:
    """Return (score, reasons) for a CoT string."""
    text = cot.lower()
    reasons: List[str] = []

    # Critical check — instant full score
    for phrase in CRITICAL_PHRASES:
        if phrase in text:
            reasons.append(f"critical:{phrase}")
            return 1.0, reasons

    # Affirmative+danger pattern
    if _AFFIRM_RE.search(cot):
        reasons.append("affirm_danger_pattern")
        return 1.0, reasons

    # Moderate accumulation
    score = 0.0
    for phrase in MODERATE_PHRASES:
        if phrase in text:
            reasons.append(f"moderate:{phrase}")
            score += 0.45

    score = min(score, 0.9)
    return score, reasons


@register("risk", "cot_parser")
class CotParserRisk:
    """Risk estimator that parses Alpamayo CoT text."""

    def __init__(
        self,
        critical_threshold: float = 1.0,
        moderate_weight: float = 0.45,
        min_score_for_log: float = 0.3,
        **_: Any,
    ) -> None:
        self._critical_threshold = critical_threshold
        self._moderate_weight = moderate_weight
        self._min_score_for_log = min_score_for_log

    def estimate(self, obs: Observation, plan: Plan) -> RiskReport:
        hint = obs.data.get("vlm_hint", {})
        cot = hint.get("cot", "")

        if not cot:
            return RiskReport(score=0.0, reasons=["no_cot"])

        score, reasons = _score_cot(cot)

        if score >= self._min_score_for_log:
            log.debug("CotParserRisk score=%.2f reasons=%s cot=%.80s", score, reasons, cot)

        return RiskReport(
            score=score,
            reasons=reasons,
            extra={"cot_snippet": cot[:120]},
        )
