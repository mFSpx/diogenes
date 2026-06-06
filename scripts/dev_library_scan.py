#!/usr/bin/env python3
"""LUCIDOTA Dev Library scanner CLI.

Usage:
  python3 scripts/dev_library_scan.py --query <topic> [--json]
  python3 scripts/dev_library_scan.py --list [--json]
  python3 scripts/dev_library_scan.py [--dry-run|--execute] [--json]

Exit codes: 0=success, 1=operational failure, 2=config/args error.

Human-facing wrapper around the legacy manifest implementation. Use this name in
new docs and operator workflows; keep the old implementation until a full
receipt-backed rename is safe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tickletrunk_scan  # noqa: E402


def main() -> int:
    # Parse --json here and pass through to tickletrunk_scan
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--json", action="store_true", help="Emit structured JSON to stdout.")
    known, remaining = ap.parse_known_args()
    # Rebuild argv for tickletrunk_scan (no --json flag there yet, so strip it)
    sys.argv = [sys.argv[0]] + remaining
    result = int(tickletrunk_scan.main())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
