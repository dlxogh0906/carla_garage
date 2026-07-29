"""Rolling ego-pose buffer for Alpamayo trajectory inference (Phase 3).

Maintains the last `num_steps` ego poses in the local frame at t0,
matching the format expected by load_physical_aiavdataset.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np
import scipy.spatial.transform as spt


class PoseBuffer:
    """Rolling window of (xyz, rotation_matrix) in world frame.

    Call `update(xyz_world, quat_wxyz)` each CARLA step.
    Call `get_local_history()` to retrieve poses in t0-local frame.
    """

    def __init__(self, num_steps: int = 16):
        self.num_steps = num_steps
        self._xyz_buf: deque = deque(maxlen=num_steps)
        self._quat_buf: deque = deque(maxlen=num_steps)  # (x, y, z, w) scipy convention

    def update(self, xyz_world: np.ndarray, quat_xyzw: np.ndarray) -> None:
        """Push current world-frame pose.

        Args:
            xyz_world: (3,) float, world-frame position in meters
            quat_xyzw: (4,) float, quaternion in (x, y, z, w) scipy convention
        """
        self._xyz_buf.append(xyz_world.astype(np.float32))
        self._quat_buf.append(quat_xyzw.astype(np.float32))

    def ready(self) -> bool:
        return len(self._xyz_buf) > 0

    def get_local_history(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (ego_history_xyz, ego_history_rot) in t0-local frame.

        Returns:
            ego_history_xyz: (16, 3) float32
            ego_history_rot: (16, 3, 3) float32
            or None if buffer is empty.

        The last entry corresponds to t0 (current step), which transforms to origin [0,0,0] + I.
        Matches the convention in load_physical_aiavdataset.py.
        """
        if not self.ready():
            return None

        xyz_list = list(self._xyz_buf)
        quat_list = list(self._quat_buf)

        # Pad if buffer not yet full
        while len(xyz_list) < self.num_steps:
            xyz_list.insert(0, xyz_list[0])
            quat_list.insert(0, quat_list[0])

        xyz_arr = np.stack(xyz_list[-self.num_steps:])    # (16, 3)
        quat_arr = np.stack(quat_list[-self.num_steps:])  # (16, 4)

        # Transform to local frame at t0 (last entry)
        t0_xyz = xyz_arr[-1].copy()
        t0_rot = spt.Rotation.from_quat(quat_arr[-1])
        t0_rot_inv = t0_rot.inv()

        xyz_local = t0_rot_inv.apply(xyz_arr - t0_xyz).astype(np.float32)
        rot_local = (t0_rot_inv * spt.Rotation.from_quat(quat_arr)).as_matrix().astype(np.float32)

        return xyz_local, rot_local
