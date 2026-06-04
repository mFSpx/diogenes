#!/usr/bin/env python3
"""Wrapper: bounded call/smoke across the local model fabric."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "scripts/lucidota_model_turbine_overseer.py", "--assign"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
