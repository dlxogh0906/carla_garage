"""Extension config: wraps upstream GlobalConfig and adds team-only fields.

Design notes:
- We never mutate `team_code/config.py`. Instead, an ExtConfig holds a
  reference to a GlobalConfig instance plus additional fields used by our
  modules (which backbones to enable, which risk estimator, etc.).
- Experiment YAMLs in `configs/` are merged on top so that swapping
  modules is a one-line change committed alongside the experiment.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExtConfig:
  # Module selections -> resolved via registry.build(kind, name).
  image_enhancer: Optional[str] = None  # e.g. "classic_cv" or "noop"
  vlm: Optional[str] = None
  risk: Optional[str] = "noop"
  safety: Optional[str] = "noop"

  # Per-module kwargs passed through to __init__.
  image_enhancer_kwargs: Dict[str, Any] = field(default_factory=dict)
  vlm_kwargs: Dict[str, Any] = field(default_factory=dict)
  risk_kwargs: Dict[str, Any] = field(default_factory=dict)
  safety_kwargs: Dict[str, Any] = field(default_factory=dict)

  # Free-form experiment metadata (owner, tag, notes).
  meta: Dict[str, Any] = field(default_factory=dict)

  # Reference to upstream GlobalConfig (set at runtime; not serialized).
  upstream: Any = None

  def apply_overlay(self, overlay: Dict[str, Any]) -> "ExtConfig":
    """Merge a YAML dict into this config in-place and return self."""
    for key, value in overlay.items():
      if not hasattr(self, key):
        # Stash unknown keys under meta so typos are discoverable but
        # don't crash experiments in progress.
        self.meta.setdefault("_unknown", {})[key] = value
        continue
      setattr(self, key, value)
    return self


def load_experiment_config(path: str | Path) -> ExtConfig:
  """Load a YAML experiment file into an ExtConfig.

    YAMLs may `extends:` another YAML (relative path) for base-overlay.
    """
  path = Path(path)
  data = _load_with_extends(path)
  cfg = ExtConfig()
  cfg.apply_overlay(data)
  return cfg


def _load_with_extends(path: Path) -> Dict[str, Any]:
  with open(path, "rt", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  base_ref = data.pop("extends", None)
  if base_ref:
    base_path = (path.parent / base_ref).resolve()
    base = _load_with_extends(base_path)
    base.update(data)
    return base
  return data
