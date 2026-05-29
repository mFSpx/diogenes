#!/usr/bin/env python3
"""Active entrypoint for INDY_READs polycareer routing and Glow Watch."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "scripts" / "legacy" / "lucidota_indy_polycareer.py"

if __name__ == "__main__":
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
