#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_attempt_class():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "attempt",
            "--text",
            "fix one small broken thing and prove it",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["verdict"] in {"PASS", "DEGRADED"}
    assert payload["attempt_engine"]["job_uuid"]
    assert payload["attempt_engine"]["receipt_path"]
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert proc.stderr == "" or "WARNING:" in proc.stderr


def test_luci_attempt_idempotency_uses_run_id_and_text():
    base = [
        str(ROOT / "luci"),
        "attempt",
        "--text",
        "fix one small broken thing and prove it",
        "--json",
    ]
    same_a = subprocess.run(base + ["--run-id", "pytest-attempt-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    same_b = subprocess.run(base + ["--run-id", "pytest-attempt-idem"], cwd=ROOT, text=True, capture_output=True, check=True)
    diff_run = subprocess.run(base + ["--run-id", "pytest-attempt-idem-2"], cwd=ROOT, text=True, capture_output=True, check=True)
    diff_text = subprocess.run(
        [
            str(ROOT / "luci"),
            "attempt",
            "--text",
            "fix one different thing and prove it",
            "--run-id",
            "pytest-attempt-idem",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload_same_a = json.loads(same_a.stdout)
    payload_same_b = json.loads(same_b.stdout)
    payload_diff_run = json.loads(diff_run.stdout)
    payload_diff_text = json.loads(diff_text.stdout)

    assert payload_same_a["attempt_engine"]["job_uuid"] == payload_same_b["attempt_engine"]["job_uuid"]
    assert payload_same_a["attempt_engine"]["receipt_path"] == payload_same_b["attempt_engine"]["receipt_path"]
    assert payload_same_a["attempt_engine"]["job_uuid"] != payload_diff_run["attempt_engine"]["job_uuid"]
    assert payload_same_a["attempt_engine"]["receipt_path"] != payload_diff_run["attempt_engine"]["receipt_path"]
    assert payload_same_a["attempt_engine"]["job_uuid"] != payload_diff_text["attempt_engine"]["job_uuid"]
    assert payload_same_a["attempt_engine"]["receipt_path"] != payload_diff_text["attempt_engine"]["receipt_path"]
