"""ActionAwareRisk — structured action-prior + trajectory disagreement risk estimator.

Priority order (highest to lowest):
  1. structured meta_action from Alpamayo JSON output
  2. trajectory-derived stop/slow intent (from pred_xyz analysis)
  3. trajectory DISAGREEMENT between TF++ path and Alpamayo pred_xyz
  4. reasoning text fallback (lightweight keyword check, last resort)

The disagreement signal (P3) captures cases where Alpamayo's expected future
motion diverges from TF++'s planned path — a physics-grounded risk cue that
doesn't require any text or meta_action parsing.

  TF++ pred_wp : (8, 2) float — 8 waypoints at 0.25s spacing -> 2.0s horizon
  TF++ checkpoint : (10, 2) float — direct-controller route checkpoints
  Alp pred_xyz : (64, 3) float — 64 waypoints at 0.1s spacing → 6.4s horizon

  Comparison points:
    wp_gru     : pred_wp[3], pred_wp[7] vs pred_xyz[9], pred_xyz[19]
    checkpoint : sample along predicted checkpoints using target_speed * t

  Forward progress gap (positive → TF++ faster, Alpamayo more conservative → risk):
    prog_gap_Xs = tf_fwd_Xs - alp_fwd_Xs

  Lateral gap (Alpamayo evading more than TF++ → potential hazard):
    lat_gap_Xs  = |alp_lat_Xs| - |tf_lat_Xs|

Config kwargs (via YAML ext_config risk_kwargs):
  meta_action_weights : dict override for meta_action → score mapping
  traj_stop_score     : float, default 0.9
  traj_slow_score     : float, default 0.55
  text_fallback_cap   : float, default 0.5
  disagree_weight     : float, blending weight for disagreement risk, default 0.5
  prog_gap_min_m      : float, minimum progress gap (m) before counting, default 1.0
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ...registry import register
from ..base import Observation, Plan, RiskReport

log = logging.getLogger(__name__)

# Structured meta_action → base risk score
_META_ACTION_SCORE: Dict[str, float] = {
    "stop":             1.0,
    "yield":            0.85,
    "slow_down":        0.6,
    "cautious_proceed": 0.35,
    "proceed":          0.0,
}

_CRITICAL_RE = re.compile(
    r"\b(hazard|out of control|collision|crash|emergency|accident)\b",
    re.IGNORECASE,
)
_MODERATE_RE = re.compile(
    r"\b(blocked|blocking|stopped in (our|the) lane|stationary (car|vehicle|object)"
    r"|caution|slow down|construction|pedestrian|obstacle|yield)\b",
    re.IGNORECASE,
)

# Normalisation constants for disagreement features
# At 40 km/h (~11 m/s), TF++ can be ~11 m ahead of a stopped Alpamayo at 1s.
# Wider range prevents premature saturation and preserves gradient.
_PROG_NORM_1S = 10.0   # max expected forward progress gap at 1s (m)
_PROG_NORM_2S = 20.0   # max expected forward progress gap at 2s (m)
_LAT_NORM_1S  = 3.0    # max expected lateral evasion gap at 1s (m)
_LAT_NORM_2S  = 5.0    # max expected lateral evasion gap at 2s (m)

# TF++ pred_wp index for each time horizon (0.25s per step, 4Hz)
_TF_IDX_1S = 3   # 4 × 0.25 s = 1.0 s
_TF_IDX_2S = 7   # 8 × 0.25 s = 2.0 s

# Alpamayo pred_xyz index for each time horizon (0.1s per step, 10Hz)
_ALP_IDX_1S = 9   # 10 × 0.1 s = 1.0 s
_ALP_IDX_2S = 19  # 20 × 0.1 s = 2.0 s


def _empty_disagreement(reason: str = "missing") -> Dict[str, Any]:
    return {
        "score": 0.0, "prog_gap_1s": 0.0, "prog_gap_2s": 0.0,
        "lat_gap_1s": 0.0, "lat_gap_2s": 0.0, "available": False,
        "reason": reason,
    }


def _text_score(text: str) -> tuple[float, List[str]]:
    if not text:
        return 0.0, []
    reasons = []
    if _CRITICAL_RE.search(text):
        reasons.append("text:critical")
        return 0.5, reasons
    hits = len(_MODERATE_RE.findall(text))
    if hits > 0:
        score = min(hits * 0.2, 0.4)
        reasons.append(f"text:moderate×{hits}")
        return score, reasons
    return 0.0, []


def _to_xy_path(path: Optional[List]) -> Optional[np.ndarray]:
    if path is None:
        return None
    try:
        arr = np.array(path, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def _sample_polyline_by_distance(points_xy: np.ndarray, distance_m: float) -> np.ndarray:
    """Sample a local-frame path at an arc distance from the ego origin."""
    origin = np.zeros((1, 2), dtype=np.float32)
    pts = np.concatenate([origin, points_xy], axis=0)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg_len.sum())
    if distance_m <= 0.0 or total <= 1e-4:
        return origin[0]

    distance_m = min(float(distance_m), total)
    cum = np.cumsum(seg_len)
    seg_idx = int(np.searchsorted(cum, distance_m, side="left"))
    if seg_idx >= len(seg_len):
        return pts[-1]
    prev_dist = 0.0 if seg_idx == 0 else float(cum[seg_idx - 1])
    denom = max(float(seg_len[seg_idx]), 1e-4)
    alpha = (distance_m - prev_dist) / denom
    return pts[seg_idx] + alpha * (pts[seg_idx + 1] - pts[seg_idx])


def _tf_points_at_horizons(
    tf_path: Optional[List],
    tf_path_kind: Optional[str],
    tf_target_speed_mps: Optional[float],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    tf_xy = _to_xy_path(tf_path)
    if tf_xy is None:
        return None, None, "missing_tf_path"

    if tf_path_kind == "checkpoint":
        if tf_target_speed_mps is None:
            return None, None, "missing_target_speed"
        speed = max(0.0, float(tf_target_speed_mps))
        return (
            _sample_polyline_by_distance(tf_xy, speed * 1.0),
            _sample_polyline_by_distance(tf_xy, speed * 2.0),
            "ok",
        )

    # Default/backward-compatible path: wp_gru waypoints at fixed 0.25s cadence.
    if tf_xy.shape[0] <= _TF_IDX_2S:
        return None, None, "short_tf_path"
    return tf_xy[_TF_IDX_1S], tf_xy[_TF_IDX_2S], "ok"


def _compute_disagreement(
    tf_pred_path: Optional[List],
    alp_pred_xyz: Optional[List],
    prog_gap_min_m: float = 1.0,
    tf_path_kind: Optional[str] = "wp_gru",
    tf_target_speed_mps: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute trajectory disagreement features between TF++ and Alpamayo.

    Returns a dict with:
      score       : float 0..1 — disagreement risk (higher = more disagreement)
      prog_gap_1s : forward progress diff at 1s (TF++ - Alp), m
      prog_gap_2s : forward progress diff at 2s (TF++ - Alp), m
      lat_gap_1s  : lateral evasion diff at 1s (|Alp| - |TF++|), m
      lat_gap_2s  : lateral evasion diff at 2s (|Alp| - |TF++|), m
      available   : bool — True if both trajectories were present
    """
    if tf_pred_path is None or alp_pred_xyz is None:
        return _empty_disagreement("missing_input")

    try:
        alp_xy = np.array(alp_pred_xyz, dtype=np.float32)  # (64, 3)
    except Exception:
        return _empty_disagreement("invalid_array")

    if alp_xy.ndim != 2 or alp_xy.shape[0] <= _ALP_IDX_2S or alp_xy.shape[1] < 2:
        return _empty_disagreement("short_alp_path")

    tf_1s, tf_2s, tf_reason = _tf_points_at_horizons(
        tf_pred_path,
        tf_path_kind,
        tf_target_speed_mps,
    )
    if tf_1s is None or tf_2s is None:
        return _empty_disagreement(tf_reason)

    # Forward progress: TF++ x-axis, Alpamayo x-axis (both ego-frame forward)
    tf_fwd_1s  = float(tf_1s[0])
    tf_fwd_2s  = float(tf_2s[0])
    alp_fwd_1s = float(alp_xy[_ALP_IDX_1S, 0])
    alp_fwd_2s = float(alp_xy[_ALP_IDX_2S, 0])

    # Lateral displacement: TF++ y-axis, Alpamayo y-axis
    tf_lat_1s  = float(tf_1s[1])
    tf_lat_2s  = float(tf_2s[1])
    alp_lat_1s = float(alp_xy[_ALP_IDX_1S, 1])
    alp_lat_2s = float(alp_xy[_ALP_IDX_2S, 1])

    prog_gap_1s = tf_fwd_1s - alp_fwd_1s   # + → TF++ faster (Alp conservative)
    prog_gap_2s = tf_fwd_2s - alp_fwd_2s
    lat_gap_1s  = abs(alp_lat_1s) - abs(tf_lat_1s)  # + → Alp more lateral (evasion)
    lat_gap_2s  = abs(alp_lat_2s) - abs(tf_lat_2s)

    # Progress risk: Alpamayo more conservative than TF++ (above minimum gap)
    prog_risk_1s = max(0.0, prog_gap_1s - prog_gap_min_m) / _PROG_NORM_1S
    prog_risk_2s = max(0.0, prog_gap_2s - prog_gap_min_m) / _PROG_NORM_2S

    # Lateral risk: Alpamayo evading significantly more than TF++
    lat_risk_1s = max(0.0, lat_gap_1s) / _LAT_NORM_1S
    lat_risk_2s = max(0.0, lat_gap_2s) / _LAT_NORM_2S

    score = min(1.0,
        0.40 * prog_risk_1s +
        0.30 * prog_risk_2s +
        0.20 * lat_risk_1s  +
        0.10 * lat_risk_2s
    )

    log.debug(
        "Disagreement: prog_1s=%.2f prog_2s=%.2f lat_1s=%.2f lat_2s=%.2f → score=%.3f",
        prog_gap_1s, prog_gap_2s, lat_gap_1s, lat_gap_2s, score,
    )

    return {
        "score":       score,
        "prog_gap_1s": round(prog_gap_1s, 3),
        "prog_gap_2s": round(prog_gap_2s, 3),
        "lat_gap_1s":  round(lat_gap_1s, 3),
        "lat_gap_2s":  round(lat_gap_2s, 3),
        "available":   True,
        "tf_path_kind": tf_path_kind or "wp_gru",
        "tf_target_speed_mps": tf_target_speed_mps,
    }


@register("risk", "action_aware")
class ActionAwareRisk:
    """Structured action-prior + trajectory-disagreement risk estimator."""

    def __init__(
        self,
        meta_action_weights: Optional[Dict[str, float]] = None,
        traj_stop_score: float = 0.9,
        traj_slow_score: float = 0.55,
        text_fallback_cap: float = 0.5,
        disagree_weight: float = 0.5,
        prog_gap_min_m: float = 1.0,
        dis_gate_threshold: float = 0.25,
        **_: Any,
    ) -> None:
        self._scores = dict(_META_ACTION_SCORE)
        if meta_action_weights:
            self._scores.update(meta_action_weights)
        self._traj_stop_score   = traj_stop_score
        self._traj_slow_score   = traj_slow_score
        self._text_fallback_cap = text_fallback_cap
        self._disagree_weight   = disagree_weight
        self._prog_gap_min_m    = prog_gap_min_m
        # Minimum dis_score before a semantic hint triggers override.
        # When dis_score < threshold TF++ already agrees with Alpamayo's intent,
        # so intervening would be redundant and causes min_speed infractions.
        self._dis_gate_threshold = max(dis_gate_threshold, 1e-6)

    def estimate(self, obs: Observation, plan: Plan) -> RiskReport:
        hint: Dict[str, Any] = obs.data.get("vlm_hint", {})

        # ── Trajectory disagreement (computed regardless of hint quality) ─
        tf_pred_path = obs.data.get("tf_pred_path")
        tf_path_kind = obs.data.get("tf_pred_path_kind")
        if tf_pred_path is None:
            tf_pred_path = obs.data.get("tf_pred_wp")
            tf_path_kind = "wp_gru" if tf_pred_path is not None else None
        tf_target_speed = obs.data.get("tf_pred_target_speed_mps")
        alp_pred_xyz = hint.get("pred_xyz") if hint else None
        dis = _compute_disagreement(
            tf_pred_path,
            alp_pred_xyz,
            self._prog_gap_min_m,
            tf_path_kind=tf_path_kind,
            tf_target_speed_mps=tf_target_speed,
        )
        dis_score = dis["score"] * self._disagree_weight

        if not hint:
            if dis["available"] and dis_score > 0.0:
                log.debug("ActionAwareRisk disagree-only score=%.3f", dis_score)
                return RiskReport(
                    score=dis_score,
                    reasons=[f"disagree:prog1s={dis['prog_gap_1s']:.1f}m"],
                    extra={"source": "disagreement_only", "disagreement": dis},
                )
            return RiskReport(score=0.0, reasons=["no_hint"])

        meta       = hint.get("meta_action")
        conf       = float(hint.get("confidence", 0.5))
        json_parsed = hint.get("json_parsed", False)

        # ── Priority 1: structured meta_action ────────────────────────────
        # Gate by trajectory disagreement: only override when TF++ genuinely
        # diverges from Alpamayo's intent. When both systems already agree
        # (dis_score low) TF++ handles the situation; intervening is redundant.
        if meta and meta in self._scores and (json_parsed or conf >= 0.60):
            base_score = self._scores[meta]
            gate = min(1.0, dis_score / self._dis_gate_threshold)
            gated_meta = base_score * conf * gate
            # dis_score alone can still contribute even if meta is weak
            score = max(gated_meta, dis_score)
            reasons = [f"meta:{meta}", f"conf:{conf:.2f}",
                       "json" if json_parsed else "keyword"]
            if dis["available"]:
                reasons.append(f"dis:{dis['score']:.2f}")
                reasons.append(f"gate:{gate:.2f}")
            log.debug(
                "ActionAwareRisk P1 meta=%s base=%.2f gate=%.2f score=%.2f",
                meta, base_score, gate, score,
            )
            return RiskReport(
                score=score,
                reasons=reasons,
                extra={
                    "source": "meta_action",
                    "meta_action": meta,
                    "confidence": conf,
                    "hazard_type": hint.get("hazard_type", "unknown"),
                    "disagreement": dis,
                    "gate": round(gate, 3),
                },
            )

        # ── Priority 2: trajectory intent (stop/slow bool) ────────────────
        # Only fire when TF++ still plans to move at meaningful speed.
        # If target_speed is already near zero the vehicle is already slowing;
        # firing P2 then creates a braking death-spiral (continuous yield → stuck).
        _P2_MIN_SPEED_MPS = 1.5  # ~5.4 km/h — ignore when already near-stopped
        traj = hint.get("traj_analysis", {})
        tf_speed_for_p2 = tf_target_speed if tf_target_speed is not None else _P2_MIN_SPEED_MPS
        if traj and tf_speed_for_p2 >= _P2_MIN_SPEED_MPS:
            gate = min(1.0, dis_score / self._dis_gate_threshold)
            if traj.get("stop_intent"):
                score = max(self._traj_stop_score * gate, dis_score)
                log.debug("ActionAwareRisk P2 traj:stop gate=%.2f dis=%.2f fwd=%.2f tf_spd=%.2f",
                          gate, dis_score, traj.get("fwd_10", 0), tf_speed_for_p2)
                return RiskReport(
                    score=score,
                    reasons=["traj:stop_intent",
                             f"fwd10={traj.get('fwd_10', 0):.2f}m",
                             f"dis:{dis['score']:.2f}", f"gate:{gate:.2f}"],
                    extra={"source": "trajectory", "traj_analysis": traj,
                           "disagreement": dis, "gate": round(gate, 3)},
                )
            if traj.get("slow_intent"):
                score = max(self._traj_slow_score * gate, dis_score)
                log.debug("ActionAwareRisk P2 traj:slow gate=%.2f dis=%.2f fwd=%.2f tf_spd=%.2f",
                          gate, dis_score, traj.get("fwd_10", 0), tf_speed_for_p2)
                return RiskReport(
                    score=score,
                    reasons=["traj:slow_intent",
                             f"fwd10={traj.get('fwd_10', 0):.2f}m",
                             f"dis:{dis['score']:.2f}", f"gate:{gate:.2f}"],
                    extra={"source": "trajectory", "traj_analysis": traj,
                           "disagreement": dis, "gate": round(gate, 3)},
                )

        # ── Priority 3: trajectory disagreement standalone ─────────────────
        if dis["available"] and dis_score > 0.0:
            log.debug("ActionAwareRisk P3 disagreement score=%.3f", dis_score)
            return RiskReport(
                score=dis_score,
                reasons=[
                    f"disagree:prog1s={dis['prog_gap_1s']:.1f}m",
                    f"prog2s={dis['prog_gap_2s']:.1f}m",
                    f"lat1s={dis['lat_gap_1s']:.2f}m",
                ],
                extra={"source": "disagreement", "disagreement": dis},
            )

        # ── Priority 4: reasoning/CoT text fallback ───────────────────────
        text = hint.get("reasoning") or hint.get("cot", "")
        score, reasons = _text_score(text)
        score = max(min(score, self._text_fallback_cap), dis_score)
        if score > 0:
            log.debug("ActionAwareRisk P4 text score=%.2f", score)
        return RiskReport(
            score=score,
            reasons=reasons or ["no_signal"],
            extra={"source": "text_fallback", "disagreement": dis},
        )
