#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_model_class():
    proc = subprocess.run(
        [str(ROOT / "luci"), "model", "admission", "--run-diogenes-gate", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["passed"] is True
    assert payload["receipt_path"]
    assert "RECEIPT_PATH=" not in proc.stdout
    assert proc.stderr == "" or proc.stderr.strip()
