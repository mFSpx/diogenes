#!/usr/bin/env python3
"""Wrapper: strict local model stack admission receipt."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "scripts/lucidota_strict_model_stack_admission.py", "--json"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
