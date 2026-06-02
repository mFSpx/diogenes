#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_model_governor_class():
    proc = subprocess.run(
        [str(ROOT / "luci"), "model", "governor", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["action_plan"]["decision"] == "defer"
    assert "allocatable_now_mb" in payload["action_plan"]


def test_luci_shell_wrapper_exposes_model_runner_validate_class():
    proc = subprocess.run(
        [str(ROOT / "luci"), "model", "runner", "validate", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
