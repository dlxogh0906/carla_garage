"""Protocol for image enhancement modules.

Input/output convention:
  - BGR uint8 numpy array
  - shape: (H, W, 3)
  - output must keep the same shape and dtype
"""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImageEnhancer(Protocol):

  def enhance(self, bgr_uint8: np.ndarray) -> np.ndarray:
    """Enhance one BGR uint8 frame."""
    ...
