#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_learning_class():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "learn",
            "--text",
            "study one current source or internal artifact, extract one reusable improvement, test it, and receipt the result",
            "--artifact",
            "scripts/dev_journey_decision_points.py",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] in {"PASS", "DEGRADED"}
    assert payload["learning_loop"]["slice"] == "luci_learning_slice"
    assert payload["learning_loop"]["promotion_decision"] in {"promote", "archive"}
    assert payload["learning_loop"]["receipt_path"]
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert proc.stderr == "" or proc.stderr.strip()
