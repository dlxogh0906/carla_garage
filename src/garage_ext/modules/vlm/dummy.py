"""Placeholder VLM implementation.

Replace with a real model. The only contract is `infer(obs) -> dict`.
"""
from typing import Any, Dict

from ...registry import register
from ..base import Observation


@register("vlm", "noop")
class NoopVLM:

    def __init__(self, **_: Any) -> None:
        pass

    def infer(self, obs: Observation) -> Dict[str, Any]:
        return {}
