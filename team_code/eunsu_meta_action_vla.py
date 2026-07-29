"""
meta_action_vla.py — Qwen2.5-VL + TTC-based Meta-action Planner

설계 원칙:
  TTC < ttc_threshold 일 때만 VLM 개입 (TF++ 정상 주행 보존)
  VLM은 8개 이산 메타-액션 중 1개 출력 → 속도 multiplier 적용

메타-액션 → 속도 multiplier:
  0 proceed           → 1.0 (TF++ 유지)
  1 slow_down         → 0.6
  2 stop              → 0.0 (brake)
  3 yield             → 0.3
  4 turn_left         → 0.7
  5 turn_right        → 0.7
  6 change_lane_left  → 0.8
  7 change_lane_right → 0.8
"""

import base64
import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ── 메타-액션 정의 ─────────────────────────────────────────────────────────────
META_ACTIONS = [
    ("proceed",           1.0),
    ("slow_down",         0.6),
    ("stop",              0.0),
    ("yield",             0.3),
    ("turn_left",         0.7),
    ("turn_right",        0.7),
    ("change_lane_left",  0.8),
    ("change_lane_right", 0.8),
]
NUM_ACTIONS = len(META_ACTIONS)

_ACTION_LIST = "\n".join(
    f"  {i}: {name} (speed x{mult})"
    for i, (name, mult) in enumerate(META_ACTIONS)
)

_PROMPT = (
    "You are a driving safety assistant for an autonomous vehicle.\n"
    "Analyze the front camera image and select the single best meta-action.\n\n"
    "Meta-actions:\n"
    f"{_ACTION_LIST}\n\n"
    "Reply with exactly one character from 0,1,2,3,4,5,6,7.\n"
    "Do not output punctuation, words, markdown, or explanations."
)

# TTC 트리거 없을 때 기본값: proceed (TF++ 유지)
_DEFAULT_ACTION_IDX = 0
_DEFAULT_MULTIPLIER = 1.0


class MetaActionVLAPlanner:
    """
    Qwen2.5-VL-7B-Instruct 기반 메타-액션 플래너.

    TTC < ttc_threshold 일 때 비동기로 VLM 추론 트리거.
    마지막 VLM 결과를 캐싱해 즉시 반환.

    Args:
        device:             torch device
        ttc_threshold:      TTC 위험 기준값 (초, 기본 3.0)
        inference_every_n:  동일 위험 상황에서 최소 재추론 간격 (스텝, 기본 20)
        model_name:         HuggingFace 모델 ID
    """

    def __init__(
        self,
        device: torch.device,
        ttc_threshold: float = 3.0,
        inference_every_n: int = 20,
        model_name: str = "/mnt/2/pretrained_models/Qwen3-VL-8B-Instruct",
        **kwargs,
    ):
        model_name = os.environ.get("META_MODEL", model_name)
        requested_device = (
            os.environ.get("META_DEVICE")
            or os.environ.get("QWEN_VLM_DEVICE")
            or str(device)
        )
        if requested_device in {"remote_vllm", "remote", "vllm_openai", "openai"}:
            self.device = device
        else:
            self.device = torch.device(requested_device)
        self.ttc_threshold = ttc_threshold
        self.inference_every_n = inference_every_n
        self.model_name = model_name
        self.backend = os.environ.get("QWEN_VLM_BACKEND", "transformers").strip().lower()
        self.remote_endpoint = os.environ.get("QWEN_VLLM_ENDPOINT", "http://127.0.0.1:8001/v1/chat/completions")
        self.remote_model = os.environ.get("QWEN_VLLM_MODEL_NAME", "").strip()
        self.remote_timeout_s = max(1.0, float(os.environ.get("QWEN_VLLM_TIMEOUT_S", "120")))
        self.max_new_tokens = max(1, int(os.environ.get("QWEN_MAX_NEW_TOKENS", "10")))
        self.enable_thinking = os.environ.get("QWEN_VLM_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.torch_dtype = self._torch_dtype_from_env()

        print(f"[MetaActionVLA] Loading {model_name} backend={self.backend} device={self.device} dtype={self.torch_dtype} ...")
        from transformers import AutoConfig, AutoProcessor  # pylint: disable=import-outside-toplevel

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = None
        if not self._is_remote_backend():
            from transformers import Qwen3VLForConditionalGeneration  # pylint: disable=import-outside-toplevel

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map={"": str(self.device)},
                trust_remote_code=True,
            )
            self.model.eval()
        print("[MetaActionVLA] Model ready.")

        # 공유 상태
        self._lock = threading.Lock()
        self._action_idx: int = _DEFAULT_ACTION_IDX
        self._multiplier: float = _DEFAULT_MULTIPLIER
        self._last_action_name: str = META_ACTIONS[_DEFAULT_ACTION_IDX][0]
        self._last_trigger_step: int = -9999
        self._worker: Optional[threading.Thread] = None
        self._last_elapsed_ms: float = 0.0
        self._total_inferences: int = 0
        self._request_seq: int = 0
        self._last_result: Dict[str, Any] = {
            "action_idx": _DEFAULT_ACTION_IDX,
            "action_name": META_ACTIONS[_DEFAULT_ACTION_IDX][0],
            "speed_multiplier": _DEFAULT_MULTIPLIER,
            "raw_response": "",
            "request_id": None,
            "request_step": None,
            "request_trigger": "",
            "benchmark": None,
        }
        self._model_info = self._collect_model_info(model_name)
        self._load_memory_info = self._collect_load_memory_info()

    # ── Public API ─────────────────────────────────────────────────────────────

    def request_guidance(self, rgb_np: np.ndarray, step: int) -> None:
        """TTC 위험 상황에서 비동기 VLM 추론 트리거. Non-blocking."""
        # 최소 재추론 간격 제한 (VLM 스팸 방지)
        if step - self._last_trigger_step < self.inference_every_n:
            return
        # 이미 추론 중이면 스킵
        if self._worker is not None and self._worker.is_alive():
            return
        self._last_trigger_step = step
        self._worker = threading.Thread(
            target=self._infer, args=(rgb_np.copy(), step), daemon=True
        )
        self._worker.start()

    def get_speed_multiplier(self) -> float:
        """현재 캐싱된 속도 multiplier 반환 (thread-safe)."""
        with self._lock:
            return self._multiplier

    def get_action_name(self) -> str:
        """현재 캐싱된 메타-액션 이름 반환 (thread-safe)."""
        with self._lock:
            return self._last_action_name

    def get_action_idx(self) -> int:
        """현재 캐싱된 메타-액션 인덱스 반환 (thread-safe)."""
        with self._lock:
            return self._action_idx

    def get_stats(self) -> dict:
        """디버그용 통계 반환."""
        with self._lock:
            return {
                "action": self._last_action_name,
                "multiplier": self._multiplier,
                "total_inferences": self._total_inferences,
                "last_elapsed_ms": self._last_elapsed_ms,
            }

    def get_last_result(self) -> Dict[str, Any]:
        with self._lock:
            out = dict(self._last_result)
            if isinstance(out.get("benchmark"), dict):
                out["benchmark"] = dict(out["benchmark"])
            return out

    # ── Internal ───────────────────────────────────────────────────────────────

    def _infer(self, rgb_np: np.ndarray, step: int) -> None:
        t0 = time.perf_counter()
        try:
            with self._lock:
                self._request_seq += 1
                request_id = self._request_seq
            if self._is_remote_backend():
                response, n_in, generated_tokens = self._infer_remote(rgb_np)
            else:
                response, n_in, generated_tokens = self._infer_transformers(rgb_np)

            action_idx = self._parse_response(response)
            action_name, multiplier = META_ACTIONS[action_idx]
            elapsed_s = time.perf_counter() - t0
            elapsed = elapsed_s * 1000
            benchmark = self._build_benchmark(
                request_id=request_id,
                step=step,
                image_np=rgb_np,
                input_tokens=n_in,
                generated_tokens=generated_tokens,
                total_s=elapsed_s,
            )

            print(
                f"[MetaActionVLA] step={step} {elapsed:.0f}ms | "
                f"action={action_name} (x{multiplier}) | raw: '{response[:20]}'"
            )

            with self._lock:
                self._action_idx = action_idx
                self._multiplier = multiplier
                self._last_action_name = action_name
                self._last_elapsed_ms = elapsed
                self._total_inferences += 1
                self._last_result = {
                    "action_idx": action_idx,
                    "action_name": action_name,
                    "speed_multiplier": multiplier,
                    "raw_response": response,
                    "request_id": request_id,
                    "request_step": step,
                    "request_trigger": "ttc",
                    "benchmark": benchmark,
                }

        except Exception as e:  # pylint: disable=broad-except
            print(f"[MetaActionVLA] inference error at step={step}: {e}")

    def _infer_transformers(self, rgb_np: np.ndarray) -> Tuple[str, int, int]:
        image = Image.fromarray(rgb_np)
        prompt = self._prompt_text()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_device = next(self.model.parameters()).device
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(model_device)
        for key, value in list(inputs.items()):
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(dtype=self.torch_dtype)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        n_in = int(inputs["input_ids"].shape[1])
        response = self.processor.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return response, n_in, int(out.shape[1] - n_in)

    def _infer_remote(self, rgb_np: np.ndarray) -> Tuple[str, int, int]:
        image = Image.fromarray(rgb_np)
        prompt = self._prompt_text()
        payload = {
            "model": self.remote_model or self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.remote_endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.remote_timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote_vllm_http_{exc.code}: {body[:500]}") from exc
        parsed = json.loads(body)
        response = str(parsed.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        usage = parsed.get("usage") if isinstance(parsed, dict) else {}
        return response, int((usage or {}).get("prompt_tokens") or 0), int((usage or {}).get("completion_tokens") or len(response))

    def _parse_response(self, text: str) -> int:
        """VLM 응답 파싱 → 메타-액션 인덱스 (0~7)."""
        stripped = text.strip()

        # 첫 문자가 숫자 0-7이면 바로 사용
        for i in range(NUM_ACTIONS):
            if stripped.startswith(str(i)):
                return i

        match = re.search(r"(?<!\d)([0-7])(?!\d)", stripped)
        if match is not None:
            return int(match.group(1))

        # 키워드 매칭 (action 이름 포함 여부)
        text_lower = text.lower()
        for i, (name, _) in enumerate(META_ACTIONS):
            keyword = name.replace("_", " ")
            if keyword in text_lower:
                return i

        print(f"[MetaActionVLA] WARNING: unparseable response '{text[:30]}', default=proceed")
        return _DEFAULT_ACTION_IDX

    def _prompt_text(self) -> str:
        if self.enable_thinking:
            return _PROMPT
        return _PROMPT + "\n/no_think"

    @staticmethod
    def _torch_dtype_from_env():
        dtype = os.environ.get("QWEN_VLM_DTYPE", "bfloat16").strip().lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        return torch.bfloat16

    def _collect_model_info(self, model_name: str) -> Dict[str, Any]:
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        checkpoint_storage_gib = 0.0
        if os.path.isdir(model_name):
            for root, _, files in os.walk(model_name):
                for name in files:
                    checkpoint_storage_gib += os.path.getsize(os.path.join(root, name))
            checkpoint_storage_gib /= 1024 ** 3
        return {
            "model": model_name,
            "model_type": type(self.model).__name__ if self.model is not None else str(getattr(self.config, "model_type", "")),
            "quant_method": os.environ.get("QWEN_QUANT", "bf16"),
            "quant_bits": self._infer_quant_bits(os.environ.get("QWEN_QUANT", ""), os.environ.get("QWEN_RUNTIME_QUANT", "")),
            "runtime_quant": os.environ.get("QWEN_RUNTIME_QUANT", "none"),
            "runtime_quantized": self._is_remote_backend() and os.environ.get("QWEN_RUNTIME_QUANT", "none") != "none",
            "param_count": param_count,
            "param_count_billion": param_count / 1e9 if param_count else None,
            "param_storage_gib": param_count * 2 / 1024 ** 3 if param_count else None,
            "checkpoint_storage_gib": checkpoint_storage_gib or (param_count * 2 / 1024 ** 3),
            "avg_bits_per_param_storage": self._infer_quant_bits(os.environ.get("QWEN_QUANT", ""), os.environ.get("QWEN_RUNTIME_QUANT", "")),
        }

    def _collect_load_memory_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "load_time_s": None,
            "delta_allocated_gib": None,
            "delta_reserved_gib": None,
            "after_allocated_gib": None,
            "after_reserved_gib": None,
        }
        if not torch.cuda.is_available():
            return info
        try:
            if self.model is None:
                info["after_reserved_gib"] = _env_float("QWEN_VLLM_AFTER_LOAD_GPU_MEM_GIB")
                before = _env_float("QWEN_VLLM_BASELINE_GPU_MEM_GIB")
                after = info["after_reserved_gib"]
                if before is not None and after is not None:
                    info["delta_reserved_gib"] = max(0.0, after - before)
                return info
            device = next(self.model.parameters()).device
            if device.type != "cuda":
                return info
            info["after_allocated_gib"] = torch.cuda.memory_allocated(device) / 1024 ** 3
            info["after_reserved_gib"] = torch.cuda.memory_reserved(device) / 1024 ** 3
        except Exception:
            pass
        return info

    def _build_benchmark(
        self,
        request_id: int,
        step: int,
        image_np: np.ndarray,
        input_tokens: int,
        generated_tokens: int,
        total_s: float,
    ) -> Dict[str, Any]:
        device = next(self.model.parameters()).device if self.model is not None else None
        model_params = int(self._model_info.get("param_count") or 0)
        decode_flops = 2 * model_params * generated_tokens if model_params else None
        bench: Dict[str, Any] = {
            "request_id": request_id,
            "model": self._model_info.get("model"),
            "model_type": self._model_info.get("model_type"),
            "quant_method": self._model_info.get("quant_method"),
            "quant_bits": self._model_info.get("quant_bits"),
            "runtime_quant": self._model_info.get("runtime_quant"),
            "runtime_quantized": self._model_info.get("runtime_quantized"),
            "device": str(device) if device is not None else f"{self.backend}:gpu{os.environ.get('QWEN_VLLM_GPU_INDEX', '')}".rstrip(":gpu"),
            "prompt_mode": "team8_meta_action_digit",
            "request_step": step,
            "request_trigger": "ttc",
            "queue_wait_s": 0.0,
            "image_h": int(image_np.shape[0]) if getattr(image_np, "ndim", 0) >= 2 else None,
            "image_w": int(image_np.shape[1]) if getattr(image_np, "ndim", 0) >= 2 else None,
            "input_tokens": int(input_tokens),
            "generated_tokens": int(generated_tokens),
            "max_new_tokens": self.max_new_tokens,
            "preprocess_latency_s": 0.0,
            "h2d_latency_s": 0.0,
            "generation_latency_s": round(total_s, 6),
            "decode_parse_latency_s": 0.0,
            "end_to_end_latency_s": round(total_s, 6),
            "tokens_per_s": round(generated_tokens / total_s, 6) if total_s > 0 else None,
            "param_count_billion": self._model_info.get("param_count_billion"),
            "param_storage_gib": self._model_info.get("param_storage_gib"),
            "checkpoint_storage_gib": self._model_info.get("checkpoint_storage_gib"),
            "avg_bits_per_param_storage": self._model_info.get("avg_bits_per_param_storage"),
            "load_time_s": self._load_memory_info.get("load_time_s"),
            "load_memory_delta_allocated_gib": self._load_memory_info.get("delta_allocated_gib"),
            "load_memory_delta_reserved_gib": self._load_memory_info.get("delta_reserved_gib"),
            "load_memory_after_allocated_gib": self._load_memory_info.get("after_allocated_gib"),
            "load_memory_after_reserved_gib": self._load_memory_info.get("after_reserved_gib"),
            "approx_decode_tflops": decode_flops / 1e12 if decode_flops else None,
            "approx_decode_tflops_per_s": (decode_flops / total_s / 1e12) if decode_flops and total_s > 0 else None,
        }
        if device is not None and device.type == "cuda":
            try:
                free, total = torch.cuda.mem_get_info(device)
                bench["cuda_allocated_gib"] = torch.cuda.memory_allocated(device) / 1024 ** 3
                bench["cuda_reserved_gib"] = torch.cuda.memory_reserved(device) / 1024 ** 3
                bench["cuda_max_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                bench["cuda_max_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024 ** 3
                bench["cuda_free_gib"] = free / 1024 ** 3
                bench["cuda_total_gib"] = total / 1024 ** 3
                after_alloc = self._load_memory_info.get("after_allocated_gib")
                after_reserved = self._load_memory_info.get("after_reserved_gib")
                if after_alloc is not None:
                    bench["generation_peak_delta_allocated_gib"] = bench["cuda_max_allocated_gib"] - after_alloc
                if after_reserved is not None:
                    bench["generation_peak_delta_reserved_gib"] = bench["cuda_max_reserved_gib"] - after_reserved
            except Exception:
                pass
        return bench

    def _is_remote_backend(self) -> bool:
        return self.backend in {"vllm", "vllm_openai", "openai", "openai_compatible", "remote"}

    @staticmethod
    def _image_data_url(pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _infer_quant_bits(*labels: str) -> Optional[int]:
        text = " ".join(str(label).lower() for label in labels if label)
        if "w8a8" in text or "int8" in text:
            return 8
        if "w4a16" in text or "awq" in text or "gptq" in text:
            return 4
        if "bf16" in text or "bfloat16" in text:
            return 16
        return None


def _env_float(name: str) -> Optional[float]:
    try:
        raw = os.environ.get(name)
        return None if raw in (None, "") else float(raw)
    except (TypeError, ValueError):
        return None
