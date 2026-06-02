#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_promptflow_batch_class(tmp_path):
    outdir = tmp_path / "promptflow_traces"
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "flow",
            "batch",
            "--dag",
            "04_RUNTIME/promptflow_smoke_flow",
            "--eval",
            "04_RUNTIME/promptflow_smoke_flow/data.jsonl",
            "--run-id",
            "wrapper_smoke",
            "--output-dir",
            str(outdir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "runtime=alive" in proc.stdout
    assert "eval_quality=unproven" in proc.stdout
    assert "receipt=" in proc.stdout or "RECEIPT_PATH=" in proc.stdout
    assert "pass_rate=" in proc.stdout


def test_luci_shell_wrapper_exposes_promptflow_batch_json_only(tmp_path):
    outdir = tmp_path / "promptflow_traces"
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "flow",
            "batch",
            "--dag",
            "04_RUNTIME/promptflow_smoke_flow",
            "--eval",
            "04_RUNTIME/promptflow_smoke_flow/data.jsonl",
            "--run-id",
            "wrapper_smoke_json",
            "--output-dir",
            str(outdir),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["runtime_status"] == "alive"
    assert payload["eval_quality"] == "unproven"
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "runtime=alive" not in proc.stdout
    assert "eval_quality=unproven" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
