#!/usr/bin/env python3
"""Wrapper: model fabric status receipt."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "scripts/goal_model_fabric_control.py", "status", "--json"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
