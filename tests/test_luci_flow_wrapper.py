#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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
        env={**os.environ, "LUCI_FLOW_DISABLE_DB_WRITE": "1"},
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
        env={**os.environ, "LUCI_FLOW_DISABLE_DB_WRITE": "1"},
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


def test_luci_shell_wrapper_exposes_visual_flow_smoke(tmp_path):
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "flow",
            "smoke",
            "--output-dir",
            str(tmp_path / "flow"),
            "--json",
        ],
        cwd=ROOT,
        env={**os.environ, "LUCI_FLOW_DISABLE_DB_WRITE": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["spec_path"].endswith(".flow.json")
    assert "flow" in payload["receipt_path"]


def test_luci_shell_wrapper_routes_slash_flow_to_visual_smoke(tmp_path):
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "/flow",
            "--smoke",
            "--output-dir",
            str(tmp_path / "flow"),
            "--json",
        ],
        cwd=ROOT,
        env={**os.environ, "LUCI_FLOW_DISABLE_DB_WRITE": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["checks"]["center_canvas"] is True


def test_luci_shell_wrapper_routes_flow_ui_alias_to_visual_smoke(tmp_path):
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "flow",
            "ui",
            "--smoke",
            "--output-dir",
            str(tmp_path / "flow"),
            "--json",
        ],
        cwd=ROOT,
        env={**os.environ, "LUCI_FLOW_DISABLE_DB_WRITE": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["checks"]["top_controls"] is True


def test_luci_shell_wrapper_accepts_flow_ui_seed_refs_in_smoke(tmp_path):
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "flow",
            "ui",
            "--flow",
            "04_RUNTIME/promptflow_smoke_flow",
            "--data",
            "04_RUNTIME/promptflow_smoke_flow/data.jsonl",
            "--smoke",
            "--output-dir",
            str(tmp_path / "flow"),
            "--json",
        ],
        cwd=ROOT,
        env={**os.environ, "LUCI_FLOW_DISABLE_DB_WRITE": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
