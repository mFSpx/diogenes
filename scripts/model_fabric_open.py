#!/usr/bin/env python3
"""Wrapper: open the local model fabric lanes."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=["needles"])
    ap.add_argument("--wait", type=int, default=35)
    args = ap.parse_args()
    cmd = [sys.executable, "scripts/goal_model_fabric_control.py", "start"]
    for target in args.target:
        cmd.extend(["--target", target])
    cmd.extend(["--wait", str(args.wait), "--json"])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
