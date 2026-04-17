"""Smoke tests: these run in CI without CARLA installed.

They verify the extension layer imports cleanly and that the registry
is wired up. Agent imports are skipped because they pull in upstream
team_code modules that depend on carla.
"""


def test_package_imports():
  import garage_ext  # noqa: F401
  from garage_ext import registry, pipeline  # noqa: F401
  from garage_ext.config import ExtConfig, load_experiment_config  # noqa: F401
  from garage_ext.modules import base  # noqa: F401


def test_builtin_modules_registered():
  import garage_ext.modules  # noqa: F401  # triggers registrations
  from garage_ext.registry import available

  assert "noop" in available("vlm")
  assert "noop" in available("risk")
  assert "noop" in available("safety")


def test_pipeline_runs_with_noop_modules():
  import garage_ext.modules  # noqa: F401
  from garage_ext.config import ExtConfig
  from garage_ext.modules.base import Control, Observation, Plan
  from garage_ext.pipeline import ExtPipeline

  cfg = ExtConfig(vlm=None, risk="noop", safety="noop")
  pipe = ExtPipeline(cfg)
  out = pipe.run(Observation(), Plan(), Control(steer=0.1, throttle=0.5, brake=0.0))
  assert out.control.steer == 0.1
  assert out.risk.score == 0.0


def test_yaml_overlay_merges():
  from pathlib import Path
  from garage_ext.config import load_experiment_config

  root = Path(__file__).resolve().parents[2]
  cfg = load_experiment_config(root / "configs" / "experiments" / "example_vlm_risk.yaml")
  assert cfg.risk == "noop"
  assert cfg.safety == "noop"
