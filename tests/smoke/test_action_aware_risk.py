import numpy as np

from garage_ext.modules.risk.action_aware_risk import _compute_disagreement


def test_disagreement_uses_direct_checkpoints_with_target_speed():
  tf_checkpoints = [[float(i), 0.0] for i in range(1, 11)]
  alp_pred_xyz = np.zeros((64, 3), dtype=np.float32).tolist()

  dis = _compute_disagreement(
      tf_checkpoints,
      alp_pred_xyz,
      tf_path_kind="checkpoint",
      tf_target_speed_mps=5.0,
  )

  assert dis["available"] is True
  assert dis["tf_path_kind"] == "checkpoint"
  assert dis["prog_gap_1s"] == 5.0
  assert dis["prog_gap_2s"] == 10.0
  assert dis["score"] > 0.0


def test_disagreement_rejects_checkpoints_without_target_speed():
  tf_checkpoints = [[float(i), 0.0] for i in range(1, 11)]
  alp_pred_xyz = np.zeros((64, 3), dtype=np.float32).tolist()

  dis = _compute_disagreement(
      tf_checkpoints,
      alp_pred_xyz,
      tf_path_kind="checkpoint",
      tf_target_speed_mps=None,
  )

  assert dis["available"] is False
  assert dis["reason"] == "missing_target_speed"


def test_disagreement_clamps_checkpoint_sample_at_path_end():
  tf_checkpoints = [[float(i), 0.0] for i in range(1, 11)]
  alp_pred_xyz = np.zeros((64, 3), dtype=np.float32).tolist()

  dis = _compute_disagreement(
      tf_checkpoints,
      alp_pred_xyz,
      tf_path_kind="checkpoint",
      tf_target_speed_mps=20.0,
  )

  assert dis["available"] is True
  assert dis["prog_gap_2s"] == 10.0
