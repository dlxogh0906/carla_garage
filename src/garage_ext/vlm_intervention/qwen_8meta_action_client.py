"""Team 8 meta-action Qwen client.

This intentionally mirrors the teammate snapshot style:
front image only, prompt asks for one digit 0-7, and the digit maps to a
speed multiplier.  It still supports the existing transformers/vLLM benchmark
plumbing so quantization runs remain comparable.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

META_ACTIONS = [
    ("proceed", 1.0),
    ("slow_down", 0.6),
    ("stop", 0.0),
    ("yield", 0.3),
    ("turn_left", 0.7),
    ("turn_right", 0.7),
    ("change_lane_left", 0.8),
    ("change_lane_right", 0.8),
]

_ACTION_LIST = "\n".join(
    f"  {i}: {name} (speed x{mult})"
    for i, (name, mult) in enumerate(META_ACTIONS)
)

PROMPT = (
    "You are a driving safety assistant for an autonomous vehicle.\n"
    "Analyze the front camera image and select the single best meta-action.\n\n"
    "Meta-actions:\n"
    f"{_ACTION_LIST}\n\n"
    "Reply with exactly one character from 0,1,2,3,4,5,6,7.\n"
    "Do not output punctuation, words, markdown, or explanations."
)

_FALLBACK: Dict[str, Any] = {
    "intervene": False,
    "risk_level": "low",
    "speed_scale": 1.0,
    "action_idx": 0,
    "action_name": "proceed",
    "raw_response": "",
    "reason": "no_result_yet",
    "request_step": None,
    "request_trigger": None,
    "prompt_mode": "team8_meta_action",
}


class Qwen8MetaActionClient:
    """Non-blocking Qwen client for 8 discrete meta-actions."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        enable_thinking: bool = False,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._enable_thinking = enable_thinking
        self._dtype_name = os.environ.get("QWEN_VLM_DTYPE", "bfloat16").lower()
        self._backend = os.environ.get("QWEN_VLM_BACKEND", "transformers").strip().lower()
        self._remote_endpoint = os.environ.get(
            "QWEN_VLLM_ENDPOINT",
            "http://127.0.0.1:8001/v1/chat/completions",
        )
        self._remote_model = os.environ.get("QWEN_VLLM_MODEL_NAME", "").strip()
        self._remote_api_key = os.environ.get("QWEN_VLLM_API_KEY", "").strip()
        self._remote_timeout_s = max(1.0, float(os.environ.get("QWEN_VLLM_TIMEOUT_S", "120")))
        self._max_new_tokens = max(1, int(os.environ.get("QWEN_MAX_NEW_TOKENS", "10")))
        self._benchmark_enabled = os.environ.get("QWEN_BENCHMARK_INFER", "1") == "1"

        self._model = None
        self._processor = None
        self._ready = False
        self._load_error: Optional[str] = None
        self._load_memory_info: Dict[str, Any] = {}
        self._model_info: Dict[str, Any] = {}

        self._req_q: queue.Queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._cached: Dict[str, Any] = dict(_FALLBACK)
        self._in_flight = False
        self._last_req_step = -999999
        self._request_seq = 0

        threading.Thread(target=self._worker, daemon=True, name="qwen-8meta").start()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def request(self, image: np.ndarray, step: int, min_step_interval: int = 20) -> bool:
        """Queue one front-image request if the client is ready and not busy."""
        if not self._ready:
            return False
        with self._result_lock:
            if self._in_flight:
                return False
        if step - self._last_req_step < min_step_interval:
            return False

        with self._result_lock:
            self._request_seq += 1
            request_id = self._request_seq

        req = {
            "request_id": request_id,
            "submit_ts": time.monotonic(),
            "image": np.asarray(image).copy(),
            "step": int(step),
        }
        try:
            self._req_q.put_nowait(req)
        except queue.Full:
            return False
        with self._result_lock:
            self._in_flight = True
        self._last_req_step = int(step)
        return True

    def get_latest(self) -> Dict[str, Any]:
        with self._result_lock:
            return dict(self._cached)

    def shutdown(self) -> None:
        try:
            self._req_q.put_nowait(None)
        except queue.Full:
            pass

    def _worker(self) -> None:
        try:
            self._load_model()
        except Exception as exc:
            self._load_error = str(exc)
            log.error("Qwen 8meta model load failed: %s", exc)
            return

        while True:
            try:
                req = self._req_q.get(timeout=2.0)
            except queue.Empty:
                continue
            if req is None:
                break
            t0 = time.perf_counter()
            try:
                result = self._infer(req, t0)
            except Exception as exc:
                log.error("Qwen 8meta inference error: %s", exc)
                result = dict(_FALLBACK) | {"reason": f"inference_error:{exc}"}
            with self._result_lock:
                self._cached = result
                self._in_flight = False
            log.info(
                "Qwen 8meta result | step=%s action=%s scale=%.2f raw=%r",
                result.get("request_step"),
                result.get("action_name"),
                result.get("speed_scale", 1.0),
                str(result.get("raw_response", ""))[:40],
            )

    def _load_model(self) -> None:
        if self._is_remote_backend():
            self._load_remote_model()
            return

        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        log.info("Loading Qwen 8meta VLM: %s (device=%s)", self._model_name, self._device)
        load_t0 = time.perf_counter()
        load_device = self._memory_probe_device(torch)
        before_mem = self._cuda_memory_snapshot(torch, load_device)

        self._processor = AutoProcessor.from_pretrained(self._model_name, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", ""))
        class_names = {
            "qwen3_vl": ["Qwen3VLForConditionalGeneration"],
            "qwen3_vl_moe": ["Qwen3VLMoeForConditionalGeneration"],
            "qwen2_5_vl": ["Qwen2_5_VLForConditionalGeneration"],
            "qwen2_vl": ["Qwen2VLForConditionalGeneration"],
        }.get(model_type, [])
        class_names += ["AutoModelForImageTextToText", "AutoModelForMultimodalLM"]

        last_error: Optional[Exception] = None
        for class_name in class_names:
            model_cls = getattr(transformers, class_name, None)
            if model_cls is None:
                continue
            try:
                self._model = model_cls.from_pretrained(
                    self._model_name,
                    torch_dtype=self._torch_dtype_arg(torch),
                    device_map=self._device_map(),
                    trust_remote_code=True,
                )
                break
            except Exception as exc:
                last_error = exc
                log.warning("Qwen 8meta class failed: %s | %s", class_name, exc)
        if self._model is None:
            raise RuntimeError(f"Could not load Qwen model_type={model_type}") from last_error

        self._model.eval()
        model_device = next(self._model.parameters()).device
        after_mem = self._cuda_memory_snapshot(torch, model_device)
        self._load_memory_info = self._build_load_memory_info(before_mem, after_mem, time.perf_counter() - load_t0)
        self._model_info = self._build_model_info(cfg)
        self._ready = True

    def _device_map(self):
        device = str(self._device)
        if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return device
        return {"": device}

    def _load_remote_model(self) -> None:
        from transformers import AutoConfig, AutoProcessor

        log.info(
            "Using remote Qwen 8meta backend: endpoint=%s model=%s",
            self._remote_endpoint,
            self._remote_model or self._model_name,
        )
        load_t0 = time.perf_counter()
        try:
            self._processor = AutoProcessor.from_pretrained(self._model_name, trust_remote_code=True)
        except Exception as exc:
            self._processor = None
            log.warning("Remote Qwen 8meta processor unavailable: %s", exc)
        cfg = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
        self._load_memory_info = {
            "load_time_s": time.perf_counter() - load_t0,
            "backend": self._backend,
            "remote_endpoint": self._remote_endpoint,
            "remote_model": self._remote_model or self._model_name,
            "before_reserved_gib": _env_float("QWEN_VLLM_BASELINE_GPU_MEM_GIB"),
            "after_reserved_gib": _env_float("QWEN_VLLM_AFTER_LOAD_GPU_MEM_GIB"),
        }
        before = self._load_memory_info.get("before_reserved_gib")
        after = self._load_memory_info.get("after_reserved_gib")
        if before is not None and after is not None:
            self._load_memory_info["delta_reserved_gib"] = max(0.0, after - before)
        self._model_info = {
            "model": self._model_name,
            "model_type": str(getattr(cfg, "model_type", "")),
            "runtime_quant": os.environ.get("QWEN_RUNTIME_QUANT", self._backend),
            "runtime_quantized": True,
            "checkpoint_storage_gib": _checkpoint_storage_gib(self._model_name),
            "load_memory": self._load_memory_info,
            "remote_backend": self._backend,
            "remote_endpoint": self._remote_endpoint,
            "remote_model": self._remote_model or self._model_name,
        }
        self._ready = True

    def _infer(self, req: dict, total_t0: float) -> Dict[str, Any]:
        from PIL import Image
        import torch

        preprocess_t0 = time.perf_counter()
        image_np = req["image"]
        pil_img = Image.fromarray(image_np.astype(np.uint8))
        if self._is_remote_backend():
            return self._infer_remote(req, pil_img, total_t0, image_np, time.perf_counter() - preprocess_t0)

        prompt_text = PROMPT + ("" if self._enable_thinking else "\n/no_think")
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt_text}]}]
        chat_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[chat_text], images=[pil_img], return_tensors="pt")
        preprocess_s = time.perf_counter() - preprocess_t0

        device = next(self._model.parameters()).device
        h2d_t0 = time.perf_counter()
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if getattr(device, "type", "") == "cuda":
            torch.cuda.synchronize(device)
        h2d_s = time.perf_counter() - h2d_t0
        input_tokens = int(inputs["input_ids"].shape[1])

        if self._benchmark_enabled and getattr(device, "type", "") == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        gen_t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        if getattr(device, "type", "") == "cuda":
            torch.cuda.synchronize(device)
        generation_s = time.perf_counter() - gen_t0

        new_tokens = out_ids[0][input_tokens:]
        decode_t0 = time.perf_counter()
        raw = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
        result = self._parse(raw)
        decode_parse_s = time.perf_counter() - decode_t0
        result["request_step"] = req.get("step")
        result["request_trigger"] = "ttc"
        result["request_id"] = req.get("request_id")
        if self._benchmark_enabled:
            result["benchmark"] = self._benchmark(
                image_np=image_np,
                input_tokens=input_tokens,
                generated_tokens=int(new_tokens.shape[0]),
                preprocess_s=preprocess_s,
                h2d_s=h2d_s,
                generation_s=generation_s,
                decode_parse_s=decode_parse_s,
                total_s=time.perf_counter() - total_t0,
                device=device,
            )
        return result

    def _infer_remote(self, req: dict, pil_img, total_t0: float, image_np: np.ndarray, preprocess_s: float) -> Dict[str, Any]:
        prompt_text = PROMPT + ("" if self._enable_thinking else "\n/no_think")
        payload = {
            "model": self._remote_model or self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_data_url(pil_img)}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": self._max_new_tokens,
            "temperature": 0,
        }
        gen_t0 = time.perf_counter()
        response = self._post_openai_chat(payload)
        generation_s = time.perf_counter() - gen_t0
        decode_t0 = time.perf_counter()
        try:
            raw = str(response["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            raw = ""
        result = self._parse(raw)
        usage = response.get("usage") if isinstance(response, dict) else {}
        input_tokens = _safe_int((usage or {}).get("prompt_tokens"), 0)
        generated_tokens = _safe_int((usage or {}).get("completion_tokens"), len(raw))
        decode_parse_s = time.perf_counter() - decode_t0
        result["request_step"] = req.get("step")
        result["request_trigger"] = "ttc"
        result["request_id"] = req.get("request_id")
        if self._benchmark_enabled:
            result["benchmark"] = self._benchmark(
                image_np=image_np,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                preprocess_s=preprocess_s,
                h2d_s=0.0,
                generation_s=generation_s,
                decode_parse_s=decode_parse_s,
                total_s=time.perf_counter() - total_t0,
                device=None,
            )
        return result

    def _parse(self, text: str) -> Dict[str, Any]:
        stripped = text.strip()
        idx = 0
        for i in range(len(META_ACTIONS)):
            if stripped.startswith(str(i)):
                idx = i
                break
        else:
            lower = text.lower()
            for i, (name, _) in enumerate(META_ACTIONS):
                if name.replace("_", " ") in lower or name in lower:
                    idx = i
                    break
        name, scale = META_ACTIONS[idx]
        return {
            "intervene": idx != 0,
            "risk_level": "low" if idx == 0 else ("critical" if scale <= 0.05 else "medium"),
            "speed_scale": float(scale),
            "action_idx": int(idx),
            "action_name": name,
            "raw_response": text[:200],
            "reason": f"team8_meta_action:{idx}:{name}",
            "prompt_mode": "team8_meta_action",
        }

    def _benchmark(
        self,
        image_np: np.ndarray,
        input_tokens: int,
        generated_tokens: int,
        preprocess_s: float,
        h2d_s: float,
        generation_s: float,
        decode_parse_s: float,
        total_s: float,
        device,
    ) -> Dict[str, Any]:
        queue_wait_s = max(0.0, total_s - (preprocess_s + h2d_s + generation_s + decode_parse_s))
        model_info = dict(self._model_info or {})
        load_memory = dict(self._load_memory_info or {})
        runtime_quant = os.environ.get("QWEN_RUNTIME_QUANT") or os.environ.get("QWEN_QUANT") or self._backend
        quant_method = os.environ.get("QWEN_QUANT") or runtime_quant
        generated_tokens = max(0, int(generated_tokens))
        tokens_per_s = generated_tokens / generation_s if generation_s > 0 and generated_tokens > 0 else None
        param_count_b = model_info.get("param_count_billion")
        param_storage_gib = model_info.get("param_storage_gib")
        checkpoint_storage_gib = model_info.get("checkpoint_storage_gib")
        avg_bits = None
        try:
            if param_count_b and checkpoint_storage_gib:
                avg_bits = float(checkpoint_storage_gib) * (1024**3) * 8.0 / (float(param_count_b) * 1e9)
        except Exception:
            avg_bits = None

        bench = {
            "backend": self._backend,
            "model": self._model_name,
            "model_type": model_info.get("model_type", ""),
            "quant_method": quant_method,
            "quant_bits": _infer_quant_bits(str(quant_method), str(runtime_quant)),
            "runtime_quant": runtime_quant,
            "runtime_quantized": model_info.get("runtime_quantized", runtime_quant not in {"none", "BF16-transformers", "BF16-vLLM"}),
            "device": str(device) if device is not None else self._backend,
            "preprocess_s": preprocess_s,
            "h2d_s": h2d_s,
            "generation_s": generation_s,
            "decode_parse_s": decode_parse_s,
            "total_s": total_s,
            "queue_wait_s": queue_wait_s,
            "preprocess_latency_s": preprocess_s,
            "h2d_latency_s": h2d_s,
            "generation_latency_s": generation_s,
            "decode_parse_latency_s": decode_parse_s,
            "end_to_end_latency_s": total_s,
            "tokens_per_s": tokens_per_s,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "max_new_tokens": self._max_new_tokens,
            "image_h": int(image_np.shape[0]) if getattr(image_np, "ndim", 0) >= 2 else None,
            "image_w": int(image_np.shape[1]) if getattr(image_np, "ndim", 0) >= 2 else None,
            "param_count_billion": param_count_b,
            "param_storage_gib": param_storage_gib,
            "checkpoint_storage_gib": checkpoint_storage_gib,
            "avg_bits_per_param_storage": avg_bits,
            "load_time_s": load_memory.get("load_time_s"),
            "load_memory_delta_allocated_gib": load_memory.get("delta_allocated_gib") or load_memory.get("delta_reserved_gib"),
            "load_memory_delta_reserved_gib": load_memory.get("delta_reserved_gib"),
            "load_memory_after_allocated_gib": load_memory.get("after_allocated_gib") or load_memory.get("after_reserved_gib"),
            "load_memory_after_reserved_gib": load_memory.get("after_reserved_gib"),
            "model_info": model_info,
            "load_memory": load_memory,
        }
        if device is not None and getattr(device, "type", "") == "cuda":
            try:
                import torch

                bench["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
                bench["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
                bench["cuda_allocated_gib"] = torch.cuda.memory_allocated(device) / 1024**3
                bench["cuda_reserved_gib"] = torch.cuda.memory_reserved(device) / 1024**3
                bench["cuda_max_allocated_gib"] = bench["peak_allocated_gib"]
                bench["cuda_max_reserved_gib"] = bench["peak_reserved_gib"]
                before_alloc = load_memory.get("after_allocated_gib")
                before_res = load_memory.get("after_reserved_gib")
                if before_alloc is not None:
                    bench["generation_peak_delta_allocated_gib"] = max(0.0, bench["peak_allocated_gib"] - float(before_alloc))
                if before_res is not None:
                    bench["generation_peak_delta_reserved_gib"] = max(0.0, bench["peak_reserved_gib"] - float(before_res))
            except Exception:
                pass
        else:
            after = bench.get("load_memory_after_allocated_gib")
            if after is not None:
                bench["cuda_max_allocated_gib"] = after
            if bench.get("load_memory_after_reserved_gib") is not None:
                bench["cuda_max_reserved_gib"] = bench.get("load_memory_after_reserved_gib")
        return bench

    def _is_remote_backend(self) -> bool:
        return self._backend in {"vllm", "vllm_openai", "openai", "openai_compatible", "remote"}

    def _torch_dtype_arg(self, torch_module):
        if self._dtype_name == "auto":
            return "auto"
        if self._dtype_name in {"bf16", "bfloat16"}:
            return torch_module.bfloat16
        if self._dtype_name in {"fp32", "float32"}:
            return torch_module.float32
        return torch_module.float16

    def _memory_probe_device(self, torch_module):
        try:
            if str(self._device).startswith("cuda:"):
                return torch_module.device(self._device)
            if str(self._device) == "cuda":
                return torch_module.device("cuda:0")
        except Exception:
            return None
        return None

    @staticmethod
    def _cuda_memory_snapshot(torch_module, device) -> Dict[str, Any]:
        if device is None or getattr(device, "type", "") != "cuda":
            return {}
        try:
            torch_module.cuda.synchronize(device)
            return {
                "device": str(device),
                "allocated_gib": torch_module.cuda.memory_allocated(device) / 1024**3,
                "reserved_gib": torch_module.cuda.memory_reserved(device) / 1024**3,
                "max_allocated_gib": torch_module.cuda.max_memory_allocated(device) / 1024**3,
                "max_reserved_gib": torch_module.cuda.max_memory_reserved(device) / 1024**3,
            }
        except Exception:
            return {}

    @staticmethod
    def _build_load_memory_info(before: Dict[str, Any], after: Dict[str, Any], load_s: float) -> Dict[str, Any]:
        info: Dict[str, Any] = {"load_time_s": load_s}
        for prefix, snap in (("before", before), ("after", after)):
            for key, value in snap.items():
                info[f"{prefix}_{key}"] = value
        for key in ("allocated_gib", "reserved_gib", "max_allocated_gib", "max_reserved_gib"):
            if key in after and key in before:
                info[f"delta_{key}"] = after[key] - before[key]
        return info

    def _build_model_info(self, cfg) -> Dict[str, Any]:
        param_count = 0
        param_bytes = 0
        dtype_counts: Dict[str, int] = {}
        try:
            for p in self._model.parameters():
                n = int(p.numel())
                dtype = str(p.dtype).replace("torch.", "")
                param_count += n
                param_bytes += int(n * p.element_size())
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + n
        except Exception:
            pass
        return {
            "model": self._model_name,
            "model_type": str(getattr(cfg, "model_type", "")),
            "runtime_quant": os.environ.get("QWEN_RUNTIME_QUANT", "none"),
            "runtime_quantized": any(("int" in k or "float8" in k) for k in dtype_counts),
            "param_count": param_count,
            "param_count_billion": param_count / 1e9 if param_count else None,
            "param_storage_bytes": param_bytes,
            "param_storage_gib": param_bytes / 1024**3 if param_bytes else None,
            "checkpoint_storage_gib": _checkpoint_storage_gib(self._model_name),
            "dtype_param_counts": dtype_counts,
            "load_memory": self._load_memory_info,
        }

    def _post_openai_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._remote_endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._remote_api_key:
            req.add_header("Authorization", f"Bearer {self._remote_api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self._remote_timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote_vllm_http_{exc.code}: {body[:500]}") from exc
        return json.loads(body)

    @staticmethod
    def _image_data_url(pil_img) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


def _checkpoint_storage_gib(model_path: str) -> Optional[float]:
    try:
        root = Path(model_path)
        if not root.exists():
            return None
        suffixes = {".bin", ".safetensors", ".pt", ".pth"}
        total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)
        return total / 1024**3 if total else None
    except Exception:
        return None


def _env_float(name: str) -> Optional[float]:
    try:
        raw = os.environ.get(name)
        return None if raw in (None, "") else float(raw)
    except (TypeError, ValueError):
        return None


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_quant_bits(*labels: str) -> Optional[int]:
    text = " ".join(label.lower() for label in labels if label)
    if "w8a8" in text or "int8" in text or "8bit" in text:
        return 8
    if "w4a16" in text or "awq" in text or "gptq" in text or "4bit" in text:
        return 4
    if "bf16" in text or "bfloat16" in text:
        return 16
    return None
