"""Traditional CV-based adaptive image enhancer.

Diagnoses each frame (low-light / over-exposed / haze / blurry / normal)
and applies the corresponding correction pipeline.
"""
from typing import Any

import cv2
import numpy as np

from ...registry import register


def _clamp01(x: float) -> float:
  return float(np.clip(x, 0.0, 1.0))


def _to_float(img_bgr: np.ndarray) -> np.ndarray:
  return img_bgr.astype(np.float32) / 255.0


def _to_uint8(img_f: np.ndarray) -> np.ndarray:
  return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)


def _analyze(img_bgr: np.ndarray):
  img_rgb_f = _to_float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
  img_u8 = img_bgr

  gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
  hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
  sat_f = hsv[:, :, 1].astype(np.float32)

  bm = float(np.mean(gray))
  bstd = float(np.std(gray))
  p10 = float(np.percentile(gray, 10))
  p90 = float(np.percentile(gray, 90))
  contrast = bstd
  sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
  sat_mean = float(np.mean(sat_f))

  min_rgb = np.min(img_rgb_f, axis=2)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
  dc_mean = float(np.mean(cv2.erode((min_rgb * 255).astype(np.uint8), kernel)))

  low_light = (0.60 * _clamp01((90.0 - bm) / 50.0) + 0.25 * _clamp01((170.0 - p90) / 80.0) + 0.15 * _clamp01(
      (45.0 - contrast) / 25.0))
  over_exp = (0.50 * _clamp01((bm - 180.0) / 50.0) + 0.30 * _clamp01((p10 - 110.0) / 60.0) + 0.20 * _clamp01(
      (45.0 - contrast) / 25.0))
  haze = (0.40 * _clamp01((45.0 - contrast) / 25.0) + 0.30 * _clamp01((70.0 - sat_mean) / 40.0) + 0.30 * _clamp01(
      (dc_mean - 90.0) / 80.0))
  blurry = _clamp01((120.0 - sharpness) / 100.0)

  scores = {"low_light": low_light, "over_exposed": over_exp, "haze": haze, "blurry": blurry}
  best = max(scores, key=scores.__getitem__)
  mode = best if scores[best] >= 0.35 else "normal"
  return mode


# --- per-mode processors (operate on BGR uint8) ---


def _gamma(img: np.ndarray, gamma: float) -> np.ndarray:
  f = _to_float(img)
  return _to_uint8(np.power(np.clip(f, 0, 1), gamma))


def _clahe(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
  return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _unsharp(img: np.ndarray, sigma: float = 1.2, amount: float = 1.2) -> np.ndarray:
  blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
  return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _compress_highlights(img: np.ndarray, strength: float = 0.7) -> np.ndarray:
  f = _to_float(img)
  bright = np.max(f, axis=2, keepdims=True)
  comp = 1.0 - strength * np.clip((bright - 0.7) / 0.3, 0, 1)
  return _to_uint8(np.clip(f * comp + f * (1.0 - comp), 0, 1))


def _contrast_stretch(img: np.ndarray) -> np.ndarray:
  out = np.zeros_like(img, dtype=np.float32)
  f = _to_float(img)
  for c in range(3):
    ch = f[:, :, c]
    lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
    out[:, :, c] = ch if (hi - lo) < 1e-6 else (ch - lo) / (hi - lo)
  return _to_uint8(np.clip(out, 0, 1))


def _increase_sat(img: np.ndarray, scale: float = 1.15) -> np.ndarray:
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
  hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
  return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _process(img: np.ndarray, mode: str) -> np.ndarray:
  if mode == "low_light":
    return _unsharp(_clahe(_gamma(img, 0.65), clip=2.5), sigma=1.0, amount=0.6)
  if mode == "over_exposed":
    return _contrast_stretch(_compress_highlights(_gamma(img, 1.35), strength=0.8))
  if mode == "haze":
    return _unsharp(_increase_sat(_clahe(_contrast_stretch(img), clip=2.0), scale=1.20), sigma=1.0, amount=0.8)
  if mode == "blurry":
    return _unsharp(img, sigma=1.2, amount=1.5)
  return img.copy()


@register("image_enhancer", "classic_cv")
class ClassicCVEnhancer:
  """Adaptive multi-mode enhancer (low-light / over-exposed / haze / blurry / normal)."""

  def __init__(self, **_: Any) -> None:
    pass

  def enhance(self, bgr_uint8: np.ndarray) -> np.ndarray:
    mode = _analyze(bgr_uint8)
    return _process(bgr_uint8, mode)


@register("image_enhancer", "noop")
class NoopEnhancer:

  def __init__(self, **_: Any) -> None:
    pass

  def enhance(self, bgr_uint8: np.ndarray) -> np.ndarray:
    return bgr_uint8
