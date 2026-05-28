#!/usr/bin/env python3
"""LUCIDOTA Dev Library scanner CLI.

Human-facing wrapper around the legacy manifest implementation. Use this name in
new docs and operator workflows; keep the old implementation until a full
receipt-backed rename is safe.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tickletrunk_scan  # noqa: E402


def main() -> int:
    return int(tickletrunk_scan.main())


if __name__ == "__main__":
    raise SystemExit(main())
