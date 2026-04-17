"""Protocol for image enhancement modules.

Receives a BGR uint8 numpy array (H, W, 3) and returns the same shape/dtype.
The enhancer is called on each camera frame before the upstream model sees it.
"""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ImageEnhancer(Protocol):
    def enhance(self, bgr_uint8: np.ndarray) -> np.ndarray:
        """Input/output: BGR uint8 (H, W, 3)."""
        ...
