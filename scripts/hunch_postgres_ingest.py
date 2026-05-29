#!/usr/bin/env python3
"""Thin wrapper for hunch Postgres ingest."""
from __future__ import annotations

import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1] / "KRAMPUSCHEWING" / "Script_Corpses"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from hunch_postgres_ingest_core import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
