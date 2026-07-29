"""Rolling frame buffer for multi-camera temporal stacking.

Maintains the last `num_frames` RGB images per camera.
Provides JPEG-encoded bytes ready for ZMQ transport.
"""

import io
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class FrameBuffer:
    """Rolling window of camera frames (BGR uint8 → RGB JPEG bytes)."""

    def __init__(self, camera_keys: List[str], camera_indices: List[int], num_frames: int = 4):
        """
        Args:
            camera_keys: input_data keys, e.g. ["rgb_front"]
            camera_indices: Alpamayo camera index for each key (e.g. [1] for front-wide)
            num_frames: rolling window length
        """
        assert len(camera_keys) == len(camera_indices)
        self.camera_keys = camera_keys
        self.camera_indices = camera_indices
        self.num_frames = num_frames
        self._bufs: Dict[str, deque] = {k: deque(maxlen=num_frames) for k in camera_keys}

    def update(self, input_data: dict) -> None:
        """Push current step's frames into the rolling buffers."""
        for key in self.camera_keys:
            if key not in input_data:
                continue
            _, frame = input_data[key]          # frame: (H, W, 4) BGRA uint8
            bgr = frame[:, :, :3]               # drop alpha
            rgb = bgr[:, :, ::-1].copy()        # BGR → RGB
            jpeg = _to_jpeg(rgb)
            self._bufs[key].append(jpeg)

    def ready(self) -> bool:
        """True when every camera has at least one frame (padded on first call)."""
        return all(len(b) > 0 for b in self._bufs.values())

    def get_frames_jpeg(self) -> Optional[List[List[bytes]]]:
        """Return (N_cams, num_frames) nested list of JPEG bytes, oldest-first.

        If a camera has fewer than num_frames buffered, pads by repeating the oldest frame.
        Returns None if no frames yet.
        """
        if not self.ready():
            return None
        result = []
        for key in self.camera_keys:
            buf = list(self._bufs[key])
            # pad left with oldest frame if buffer not yet full
            while len(buf) < self.num_frames:
                buf.insert(0, buf[0])
            result.append(buf[-self.num_frames:])
        return result

    def get_camera_indices(self) -> List[int]:
        return list(self.camera_indices)


def _to_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
