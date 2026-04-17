"""Tiny string->class registry used to swap modules from YAML configs.

Usage:
    from garage_ext.registry import register, build

    @register("risk", "heuristic_v1")
    class HeuristicRisk: ...

    risk = build("risk", "heuristic_v1", cfg=...)

The registry keeps module implementations decoupled: a teammate can add a
new estimator without touching pipeline code or existing modules.
"""
from typing import Any, Callable, Dict, Type

_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {}


def register(kind: str, name: str) -> Callable[[Type[Any]], Type[Any]]:
  """Decorator: register a class under (kind, name)."""

  def _wrap(cls: Type[Any]) -> Type[Any]:
    bucket = _REGISTRY.setdefault(kind, {})
    if name in bucket and bucket[name] is not cls:
      raise ValueError(f"Duplicate registration: {kind}/{name}")
    bucket[name] = cls
    return cls

  return _wrap


def build(kind: str, name: str, **kwargs: Any) -> Any:
  """Instantiate a registered class by (kind, name)."""
  try:
    cls = _REGISTRY[kind][name]
  except KeyError as e:
    available = list(_REGISTRY.get(kind, {}).keys())
    raise KeyError(f"No {kind!r} named {name!r}. Available: {available}") from e
  return cls(**kwargs)


def available(kind: str) -> list:
  return sorted(_REGISTRY.get(kind, {}).keys())
