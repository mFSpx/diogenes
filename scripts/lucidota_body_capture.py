#!/usr/bin/env python3
"""Active/importable entrypoint for Body Capture until the Rust adapter replaces it."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts" / "legacy" / "lucidota_body_capture.py"

_spec = importlib.util.spec_from_file_location("_lucidota_legacy_body_capture", IMPLEMENTATION)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {IMPLEMENTATION}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

for _name, _value in vars(_module).items():
    if not _name.startswith("__") or _name == "__doc__":
        globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(_module.main())
