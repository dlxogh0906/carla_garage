"""
meta_action_vla.py â Qwen2.5-VL + TTC-based Meta-action Planner

ì¤ê³ ìì¹:
  TTC < ttc_threshold ì¼ ëë§ VLM ê°ì (TF++ ì ì ì£¼í ë³´ì¡´)
  VLMì 8ê° ì´ì° ë©í-ì¡ì ì¤ 1ê° ì¶ë ¥ â ìë multiplier ì ì©

ë©í-ì¡ì â ìë multiplier:
  0 proceed           â 1.0 (TF++ ì ì§)
  1 slow_down         â 0.6
  2 stop              â 0.0 (brake)
  3 yield             â 0.3
  4 turn_left         â 0.7
  5 turn_right        â 0.7
  6 change_lane_left  â 0.8
  7 change_lane_right â 0.8
"""

import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ââ ë©í-ì¡ì ì ì âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
META_ACTIONS = [
    ("proceed",           1.0),
    ("cautious_proceed",  0.85),
    ("slow_down",         0.6),
    ("yield",             0.35),
    ("crawl",             0.2),
    ("hard_brake",        0.1),
    ("stop",              0.0),
    ("emergency_stop",    0.0),
]
NUM_ACTIONS = len(META_ACTIONS)

_ACTION_LIST = "\n".join(
    f"  {i}: {name} (speed x{mult})"
    for i, (name, mult) in enumerate(META_ACTIONS)
)

_PROMPT_TEMPLATE = """\
You are a speed-control meta-action planner for an autonomous vehicle.

You receive:
- An annotated front RGB image. The yellow overlay is an approximate ego driving corridor.
- Compact scalar driving context below.

Available meta-actions:
{action_list}

Current driving situation:
- Ego speed: {ego_speed_text}
- Distance to nearest front object: {front_distance_text}
- Sensitive TTC estimate: {ttc_text}
- TTC source: {ttc_source}
- TF++ target speed before intervention: {tfpp_target_speed_text}

Choose exactly one speed-control meta-action. Do not suggest steering, lane changes, or route changes.

Respond ONLY with one JSON object. No markdown, no extra text:
{{
  "action_id": <integer 0-7>,
  "action": "<one of the action names>",
  "risk_level": "<low|medium|high|critical>",
  "speed_scale": <float 0.0 to 1.0>,
  "primary_hazard_id": <integer or null>,
  "hazard_type": "<none|vehicle|pedestrian|traffic_light|stop_sign|obstacle|unknown>",
  "path_blocked": <true or false>,
  "rule_type": "<none|red_light|yellow_light|stop_sign|unknown>",
  "reason": "<one brief sentence>"
}}

Rules:
- Default to proceed. Preserve TF++ unless a visible object physically blocks or is entering the ego corridor.
- Do not reduce speed for side-lane, oncoming, parked, or distant vehicles that are outside the yellow ego corridor.
- If a visible vehicle, pedestrian, cyclist, motorcycle, or obstacle blocks or is entering the ego corridor, reduce speed.
- If sensitive TTC < 1.0 s for a same-lane hazard, choose hard_brake, stop, or emergency_stop.
- If sensitive TTC is 1.0-2.0 s and the corridor is blocked, choose yield, crawl, or hard_brake.
- If sensitive TTC is 2.0-3.0 s and the corridor is blocked, choose cautious_proceed or slow_down.
- If a relevant red/yellow traffic light or stop sign is visible for the ego lane/route, choose stop.
- Do not stop for irrelevant side/behind traffic lights, already-passed signs, or hazards outside the ego corridor.
- speed_scale should match the selected action scale unless there is a strong reason to be more conservative.
"""

_TRAFFIC_RULE_PROMPT_TEMPLATE = """\
You are a traffic-rule critic for an autonomous vehicle.

Your only task is to decide whether the ego vehicle must stop or stay stopped for:
- a red or yellow traffic light relevant to the ego lane/route
- a stop sign relevant to the ego lane/route

Do NOT intervene for generic vehicles, pedestrians, lane changes, or obstacles in this prompt.

You receive:
- An annotated front RGB image. The yellow overlay is an approximate ego driving corridor.
- TF++ traffic-rule candidates below. These candidates can be noisy; verify relevance and state from the image.
- Minimal driving context below.

Current driving situation:
- Ego speed: {ego_speed_text}
- Distance to nearest front object: {front_distance_text}
- Sensitive TTC estimate: {ttc_text}
- TTC source: {ttc_source}
- TF++ target speed before intervention: {tfpp_target_speed_text}

Traffic-rule candidates from TF++:
{rule_context}

Respond ONLY with one JSON object. No markdown, no extra text:
{{
  "rule_intervene": <true or false>,
  "rule_type": "<none|red_light|yellow_light|stop_sign|unknown>",
  "traffic_light_state": "<red|yellow|green|unknown|not_visible>",
  "stop_sign_visible": <true or false>,
  "relevant_to_ego": <true or false>,
  "confidence": <float 0.0 to 1.0>,
  "speed_scale": <float 0.0 to 1.0>,
  "reason": "<one brief sentence>"
}}

Rules:
- Only intervene if the rule object appears relevant to the ego lane/route.
- Relevant red light -> rule_intervene true, rule_type "red_light", speed_scale 0.0.
- Relevant yellow light -> rule_intervene true, rule_type "yellow_light", speed_scale 0.0.
- Relevant stop sign before the ego reaches it -> rule_intervene true, rule_type "stop_sign", speed_scale 0.0.
- Relevant green traffic light -> rule_intervene false, traffic_light_state "green", relevant_to_ego true, confidence >= 0.75, speed_scale 1.0.
- Irrelevant side/behind traffic lights, already-passed signs, or tiny uncertain signals -> rule_intervene false.
- If you cannot see a traffic light in the image, use traffic_light_state "not_visible", relevant_to_ego false, confidence 0.0.
- Use traffic_light_state "unknown" only when a signal is visible but its color cannot be determined.
- If confidence < 0.70, rule_intervene must be false.
"""

_GAP_PROMPT_TEMPLATE = """\
You are a non-signalized intersection gap critic for an autonomous vehicle.

Your job is to decide whether the ego vehicle should enter a junction now.
Be practical: only slow or wait for actors that are already crossing the ego
path or will reach the conflict zone immediately. Merely visible side/oncoming
traffic is not enough.

You receive:
- An annotated front RGB image. The yellow overlay is the approximate ego driving corridor.
- A BEV inset. Magenta marks crossing/oncoming gap candidates, red marks the primary front hazard, yellow marks ego-lane objects.
- Compact route context below.

Available meta-actions:
{action_list}

Current driving situation:
- Ego speed: {ego_speed_text}
- Distance to nearest front object: {front_distance_text}
- Sensitive TTC estimate: {ttc_text}
- TTC source: {ttc_source}
- TF++ target speed before intervention: {tfpp_target_speed_text}

Intersection gap context:
{gap_context}

Respond ONLY with one JSON object. No markdown, no extra text:
{{
  "action_id": <integer 0-7>,
  "action": "<one of the action names>",
  "gap_decision": "<go|cautious_go|creep|yield|stop|unknown>",
  "clear_to_enter": <true or false>,
  "cross_traffic": <true or false>,
  "confidence": <float 0.0 to 1.0>,
  "risk_level": "<low|medium|high|critical>",
  "speed_scale": <float 0.0 to 1.0>,
  "primary_hazard_id": <integer or null>,
  "hazard_type": "<none|vehicle|pedestrian|emergency_vehicle|obstacle|unknown>",
  "path_blocked": <true or false>,
  "reason": "<one brief sentence>"
}}

Rules:
- This is for unsignalized junction entry timing, not traffic-light compliance.
- Default to proceed or cautious_proceed when the planned path is visually clear.
- If a candidate is parked, waiting in its own lane, behind ego, far from the conflict point, or not moving toward the ego path, choose proceed or cautious_proceed.
- Choose slow_down or yield only when a vehicle/pedestrian is already entering or crossing the planned path.
- Choose stop only for an immediate collision risk in the conflict zone. Do not stop just because cross traffic is visible.
- If the ego is already stopped and the conflict zone is no longer occupied, choose cautious_proceed.
- Do not suggest steering, lane changes, or route changes; only choose a speed-control meta-action.
- If confidence < 0.70, choose cautious_proceed rather than stop/yield.
"""

# TTC í¸ë¦¬ê±° ìì ë ê¸°ë³¸ê°: proceed (TF++ ì ì§)
_DEFAULT_ACTION_IDX = 0
_DEFAULT_MULTIPLIER = 1.0


class MetaActionVLAPlanner:
    """
    Qwen2.5-VL-7B-Instruct ê¸°ë° ë©í-ì¡ì íëë.

    TTC < ttc_threshold ì¼ ë ë¹ëê¸°ë¡ VLM ì¶ë¡  í¸ë¦¬ê±°.
    ë§ì§ë§ VLM ê²°ê³¼ë¥¼ ìºì±í´ ì¦ì ë°í.

    Args:
        device:             torch device
        ttc_threshold:      TTC ìí ê¸°ì¤ê° (ì´, ê¸°ë³¸ 3.0)
        inference_every_n:  ëì¼ ìí ìí©ìì ìµì ì¬ì¶ë¡  ê°ê²© (ì¤í, ê¸°ë³¸ 20)
        model_name:         HuggingFace ëª¨ë¸ ID
    """

    def __init__(
        self,
        device: torch.device,
        ttc_threshold: float = 3.0,
        inference_every_n: int = 20,
        model_name: str = "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct",
        **kwargs,
    ):
        self.device = device
        self.ttc_threshold = ttc_threshold
        self.inference_every_n = inference_every_n
        self.rule_inference_every_n = max(
            1,
            int(os.environ.get("META_RULE_EVERY_N_STEPS", max(1, min(5, inference_every_n)))),
        )
        self.gap_inference_every_n = max(
            1,
            int(os.environ.get("META_GAP_EVERY_N_STEPS", max(1, min(5, inference_every_n)))),
        )

        print(f"[MetaActionVLA] Loading {model_name} ...")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # pylint: disable=import-outside-toplevel

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # TF++가 cuda:0을 점유하므로 Qwen은 cuda:1에 단독 배치 (내 qwen_client와 동일 방식)
        vlm_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=vlm_device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("[MetaActionVLA] Model ready.")

        # ê³µì  ìí
        self._lock = threading.Lock()
        self._action_idx: int = _DEFAULT_ACTION_IDX
        self._multiplier: float = _DEFAULT_MULTIPLIER
        self._last_action_name: str = META_ACTIONS[_DEFAULT_ACTION_IDX][0]
        self._last_trigger_step: int = -9999
        self._last_trigger_step_by_mode: Dict[str, int] = {
            "speed": -9999,
            "traffic_rule": -9999,
            "gap": -9999,
        }
        self._worker: Optional[threading.Thread] = None
        self._last_elapsed_ms: float = 0.0
        self._total_inferences: int = 0
        self._last_reason: str = ""
        self._last_risk_level: str = "low"
        self._last_raw_response: str = ""
        self._last_result: Dict[str, Any] = self._fallback_result()

    # ââ Public API âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def request_guidance(
        self,
        rgb_np: np.ndarray,
        step: int,
        context: Optional[Dict[str, Any]] = None,
        prompt_mode: str = "speed",
    ) -> bool:
        """TTC ìí ìí©ìì ë¹ëê¸° VLM ì¶ë¡  í¸ë¦¬ê±°. Non-blocking."""
        # ìµì ì¬ì¶ë¡  ê°ê²© ì í (VLM ì¤í¸ ë°©ì§)
        if prompt_mode not in ("speed", "traffic_rule", "gap"):
            prompt_mode = "speed"
        if prompt_mode == "traffic_rule":
            min_gap = self.rule_inference_every_n
        elif prompt_mode == "gap":
            min_gap = self.gap_inference_every_n
        else:
            min_gap = self.inference_every_n
        if step - self._last_trigger_step_by_mode.get(prompt_mode, -9999) < min_gap:
            return False
        # ì´ë¯¸ ì¶ë¡  ì¤ì´ë©´ ì¤íµ
        if self._worker is not None and self._worker.is_alive():
            return False
        self._last_trigger_step = step
        self._last_trigger_step_by_mode[prompt_mode] = step
        self._worker = threading.Thread(
            target=self._infer, args=(rgb_np.copy(), step, dict(context or {}), prompt_mode), daemon=True
        )
        self._worker.start()
        return True

    def get_speed_multiplier(self) -> float:
        """íì¬ ìºì±ë ìë multiplier ë°í (thread-safe)."""
        with self._lock:
            return self._multiplier

    def get_action_name(self) -> str:
        """íì¬ ìºì±ë ë©í-ì¡ì ì´ë¦ ë°í (thread-safe)."""
        with self._lock:
            return self._last_action_name

    def get_action_idx(self) -> int:
        """íì¬ ìºì±ë ë©í-ì¡ì ì¸ë±ì¤ ë°í (thread-safe)."""
        with self._lock:
            return self._action_idx

    def get_latest_result(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_result)

    def get_stats(self) -> dict:
        """ëë²ê·¸ì© íµê³ ë°í."""
        with self._lock:
            return {
                "action": self._last_action_name,
                "multiplier": self._multiplier,
                "total_inferences": self._total_inferences,
                "last_elapsed_ms": self._last_elapsed_ms,
                "risk_level": self._last_risk_level,
                "reason": self._last_reason,
                "raw_response": self._last_raw_response,
            }

    # ââ Internal âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _infer(self, rgb_np: np.ndarray, step: int, context: Dict[str, Any], prompt_mode: str) -> None:
        t0 = time.perf_counter()
        try:
            image = Image.fromarray(rgb_np)
            if prompt_mode == "traffic_rule":
                prompt_template = _TRAFFIC_RULE_PROMPT_TEMPLATE
            elif prompt_mode == "gap":
                prompt_template = _GAP_PROMPT_TEMPLATE
            else:
                prompt_template = _PROMPT_TEMPLATE
            prompt = prompt_template.format(
                action_list=_ACTION_LIST,
                ego_speed_text=_format_speed(context.get("ego_speed")),
                front_distance_text=_format_metric(context.get("front_distance"), "m"),
                ttc_text=_format_metric(context.get("ttc"), "s"),
                ttc_source=str(context.get("ttc_source", "unknown")),
                tfpp_target_speed_text=_format_speed(context.get("tfpp_target_speed")),
                object_table=context.get("object_table", "No detected objects available."),
                rule_context=context.get("rule_context", "No TF++ rule-object summary available."),
                gap_context=context.get("gap_context", "No intersection gap context available."),
                path_summary=context.get("path_summary", "TF++ path checkpoints unavailable."),
                ttc_history=context.get("ttc_history", "No history yet."),
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt + "\n/no_think"},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            n_in = inputs["input_ids"].shape[1]
            response = self.processor.decode(
                out[0][n_in:], skip_special_tokens=True
            ).strip()

            action_idx, parsed = self._parse_response(response, prompt_mode=prompt_mode)
            action_name, default_multiplier = META_ACTIONS[action_idx]
            multiplier = min(default_multiplier, _clamp01(parsed.get("speed_scale", default_multiplier), default_multiplier))
            elapsed = (time.perf_counter() - t0) * 1000
            risk_level = str(parsed.get("risk_level", "low"))[:20]
            reason = str(parsed.get("reason", ""))[:120]

            print(
                f"[MetaActionVLA] step={step} {elapsed:.0f}ms | "
                f"action={action_name} (x{multiplier:.2f}) risk={risk_level} | raw: '{response[:120]}'"
            )

            with self._lock:
                self._action_idx = action_idx
                self._multiplier = multiplier
                self._last_action_name = action_name
                self._last_elapsed_ms = elapsed
                self._total_inferences += 1
                self._last_reason = reason
                self._last_risk_level = risk_level
                self._last_raw_response = response[:200]
                self._last_result = dict(parsed) | {
                    "action_id": action_idx,
                    "action": action_name,
                    "intervene": bool(parsed.get("intervene", multiplier < 0.95)),
                    "speed_scale": multiplier,
                    "risk_level": risk_level,
                    "reason": reason,
                    "raw_response": response[:200],
                    "request_step": step,
                    "prompt_mode": prompt_mode,
                }

        except Exception as e:  # pylint: disable=broad-except
            print(f"[MetaActionVLA] inference error at step={step}: {e}")

    def _parse_response(self, text: str, prompt_mode: str = "speed") -> Tuple[int, Dict[str, Any]]:
        """VLM ìëµ íì± â ë©í-ì¡ì ì¸ë±ì¤ (0~7)."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if prompt_mode == "traffic_rule":
                    return _parse_rule_response(data)

                action_idx = _coerce_action_idx(data)
                if action_idx is not None:
                    data["speed_scale"] = min(META_ACTIONS[action_idx][1], _clamp01(
                        data.get("speed_scale", META_ACTIONS[action_idx][1]),
                        META_ACTIONS[action_idx][1],
                    ))
                    data["risk_level"] = _coerce_risk_level(data.get("risk_level"))
                    data["reason"] = str(data.get("reason", ""))[:120]
                    if prompt_mode == "gap":
                        data["gap_decision"] = _coerce_gap_decision(data.get("gap_decision"))
                        data["clear_to_enter"] = bool(data.get("clear_to_enter", data["speed_scale"] >= 0.95))
                        data["cross_traffic"] = bool(data.get("cross_traffic", data["speed_scale"] < 0.95))
                        data["gap_confidence"] = _clamp01(data.get("confidence", data.get("gap_confidence", 0.0)), 0.0)
                    return action_idx, data
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"[MetaActionVLA] JSON parse warning: {exc} | raw='{cleaned[:120]}'")

        stripped = cleaned.strip()

        # ì²« ë¬¸ìê° ì«ì 0-7ì´ë©´ ë°ë¡ ì¬ì©
        for i in range(NUM_ACTIONS):
            if stripped.startswith(str(i)):
                return i, {"speed_scale": META_ACTIONS[i][1], "risk_level": "unknown", "reason": "legacy digit response"}

        # í¤ìë ë§¤ì¹­ (action ì´ë¦ í¬í¨ ì¬ë¶)
        text_lower = cleaned.lower()
        for i, (name, _) in enumerate(META_ACTIONS):
            if name in text_lower or name.replace("_", " ") in text_lower:
                return i, {"speed_scale": META_ACTIONS[i][1], "risk_level": "unknown", "reason": "legacy keyword response"}

        print(f"[MetaActionVLA] WARNING: unparseable response '{cleaned[:120]}', default=proceed")
        return _DEFAULT_ACTION_IDX, {
            "speed_scale": _DEFAULT_MULTIPLIER,
            "risk_level": "parse_fail",
            "reason": "parse_fail",
        }

    @staticmethod
    def _fallback_result() -> Dict[str, Any]:
        return {
            "action_id": _DEFAULT_ACTION_IDX,
            "action": META_ACTIONS[_DEFAULT_ACTION_IDX][0],
            "speed_scale": _DEFAULT_MULTIPLIER,
            "risk_level": "low",
            "reason": "no_result_yet",
            "raw_response": "",
            "request_step": None,
            "prompt_mode": "speed",
            "rule_intervene": False,
            "rule_type": "none",
            "traffic_light_state": "unknown",
            "stop_sign_visible": False,
            "relevant_to_ego": False,
            "rule_confidence": 0.0,
            "rule_speed_scale": 1.0,
            "rule_reason": "",
            "gap_decision": "unknown",
            "clear_to_enter": True,
            "cross_traffic": False,
            "gap_confidence": 0.0,
        }


def _coerce_action_idx(data: Dict[str, Any]) -> Optional[int]:
    raw_id = data.get("action_id", data.get("action_idx", data.get("action")))
    try:
        action_idx = int(raw_id)
        if 0 <= action_idx < NUM_ACTIONS:
            return action_idx
    except (TypeError, ValueError):
        pass

    raw_action = str(data.get("action", "")).strip().lower().replace(" ", "_")
    for idx, (name, _) in enumerate(META_ACTIONS):
        if raw_action == name:
            return idx
    return None


def _parse_rule_response(data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    rule_type = str(data.get("rule_type", "none")).lower()
    if rule_type not in {"none", "red_light", "yellow_light", "stop_sign", "unknown"}:
        rule_type = "unknown"

    light_state = str(data.get("traffic_light_state", "unknown")).lower()
    if light_state not in {"red", "yellow", "green", "unknown", "not_visible", "none"}:
        light_state = "unknown"

    confidence = _clamp01(data.get("confidence", data.get("rule_confidence", 0.0)), 0.0)
    relevant = bool(data.get("relevant_to_ego", False))
    rule_intervene = bool(data.get("rule_intervene", data.get("intervene", False)))
    stop_visible = bool(data.get("stop_sign_visible", rule_type == "stop_sign"))
    rule_scale = _clamp01(data.get("speed_scale", data.get("rule_speed_scale", 1.0)), 1.0)

    if confidence < 0.70 or not relevant:
        rule_intervene = False

    if not rule_intervene:
        action_idx = 0
        rule_scale = 1.0
        risk_level = "low"
    elif rule_type in {"red_light", "yellow_light", "stop_sign"}:
        action_idx = 6
        rule_scale = 0.0
        risk_level = "critical"
    else:
        action_idx = 2
        rule_scale = min(rule_scale, 0.6)
        risk_level = "medium"

    reason = str(data.get("reason", ""))[:120]
    parsed = {
        "intervene": rule_intervene,
        "speed_scale": rule_scale,
        "risk_level": risk_level,
        "primary_hazard_id": data.get("primary_hazard_id"),
        "hazard_type": rule_type,
        "path_blocked": False,
        "tfpp_plan_safe": not rule_intervene,
        "rule_intervene": rule_intervene,
        "rule_type": rule_type,
        "traffic_light_state": light_state,
        "stop_sign_visible": stop_visible,
        "relevant_to_ego": relevant,
        "rule_confidence": confidence,
        "rule_speed_scale": rule_scale,
        "rule_reason": reason,
        "reason": reason,
        "prompt_mode": "traffic_rule",
    }
    return action_idx, parsed


def _coerce_risk_level(value: Any) -> str:
    risk = str(value or "low").lower()
    return risk if risk in {"low", "medium", "high", "critical"} else "low"


def _coerce_gap_decision(value: Any) -> str:
    decision = str(value or "unknown").lower().replace(" ", "_")
    return decision if decision in {"go", "cautious_go", "creep", "yield", "stop", "unknown"} else "unknown"


def _format_metric(value: Any, unit: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not available"
    if not math.isfinite(numeric) or numeric >= 999:
        return "not available"
    return f"{numeric:.2f} {unit}"


def _format_speed(value: Any) -> str:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return "not available"
    if not math.isfinite(speed):
        return "not available"
    return f"{speed:.2f} m/s ({speed * 3.6:.1f} km/h)"


def _clamp01(value: Any, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return max(0.0, min(1.0, numeric))
