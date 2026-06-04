#!/usr/bin/env python3
"""Wrapper: release/stop local model fabric lanes."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=["optional"])
    args = ap.parse_args()
    cmd = [sys.executable, "scripts/goal_model_fabric_control.py", "stop"]
    for target in args.target:
        cmd.extend(["--target", target])
    cmd.append("--json")
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
