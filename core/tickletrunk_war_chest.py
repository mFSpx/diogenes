#!/usr/bin/env python3
"""TickleTrunk war chest helpers for mutation-index-test loops."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TestReceipt:
    ok: bool
    returncode: int
    stdout: str
    stderr: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_math_pytest() -> TestReceipt:
    cp = subprocess.run(
        [sys.executable, "-m", "pytest", "math/", "-v"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        cwd=str(ROOT),
    )
    return TestReceipt(ok=cp.returncode == 0, returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)


def rerun_tickletrunk_scan() -> TestReceipt:
    cp = subprocess.run(
        [sys.executable, "scripts/tickletrunk_scan.py", "--execute"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        cwd=str(ROOT),
    )
    return TestReceipt(ok=cp.returncode == 0, returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)
