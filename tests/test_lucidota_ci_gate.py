#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.lucidota_ci_gate as ci_gate

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/lucidota_ci_gate.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )


def test_ci_gate_help_does_not_execute_long_gate() -> None:
    proc = run("--help")
    assert proc.returncode == 0
    assert "Run the local LUCIDOTA CI gate" in proc.stdout
    assert "REPORT_PATH=" not in proc.stdout


def test_ci_gate_lists_steps_without_executing() -> None:
    proc = run("--list-steps")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.ci_gate.plan.v1"
    assert any("lucidota_mega_gate.py" in step for step in payload["steps"])
    assert any("lucidota_strict_model_stack_admission.py" in step for step in payload["steps"])
    assert any("quality_work_order_compiler.py" in step for step in payload["steps"])
    assert "REPORT_PATH=" not in proc.stdout


def test_ci_gate_captures_receipt_path_from_strict_admission(monkeypatch) -> None:
    class P:
        returncode = 0
        stdout = "STRICT_MODEL_STACK_ADMISSION=PASS\nRECEIPT_PATH=05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: P())
    step = ci_gate.run([sys.executable, "scripts/lucidota_strict_model_stack_admission.py", "--json"], timeout_sec=5)
    assert step["report_path"] == "05_OUTPUTS/model_runtime/strict_model_stack_admission_latest.json"
