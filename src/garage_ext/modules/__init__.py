"""Module implementations live in vlm/, risk/, safety/ subpackages.

Importing this package forces each subpackage to register its built-in
implementations into the global registry.
"""
import importlib
import logging

_log = logging.getLogger(__name__)

for _name in ("vlm", "risk", "safety", "image_enhancer"):
  try:
    importlib.import_module(f"{__name__}.{_name}")
  except ModuleNotFoundError as exc:
    _log.warning("Skipping optional garage_ext module %s: %s", _name, exc)
