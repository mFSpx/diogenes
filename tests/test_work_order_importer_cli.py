#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def json_line(stdout: str) -> dict:
    return next(json.loads(line) for line in stdout.splitlines() if line.startswith("{"))


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=10)


def test_work_order_importer_pipeline_dry_run_does_not_create_queue_state(tmp_path: Path) -> None:
    src = tmp_path / "drop"
    src.mkdir()
    (src / "note.md").write_text("Alice saw Evidence.", encoding="utf-8")
    absurd = tmp_path / "absurd"
    proc = run([
        sys.executable,
        "scripts/work_order_importer.py",
        "pipeline",
        "--case-id",
        "cli-case",
        "--source-folder",
        str(src),
        "--absurd-dir",
        str(absurd),
        "--json",
    ])
    assert proc.returncode == 0, proc.stderr
    payload = json_line(proc.stdout)
    assert payload["status"] == "DRY_RUN"
    assert payload["execute_performed"] is False
    assert payload["job_count"] == 6
    assert not (absurd / "jobs_state.json").exists()


def test_work_order_importer_pipeline_execute_writes_absurd_queue_state(tmp_path: Path) -> None:
    src = tmp_path / "drop"
    src.mkdir()
    (src / "note.md").write_text("Alice saw Evidence.", encoding="utf-8")
    absurd = tmp_path / "absurd"
    proc = run([
        sys.executable,
        "scripts/work_order_importer.py",
        "pipeline",
        "--case-id",
        "cli-case",
        "--source-folder",
        str(src),
        "--absurd-dir",
        str(absurd),
        "--execute",
        "--json",
    ])
    assert proc.returncode == 0, proc.stderr
    payload = json_line(proc.stdout)
    assert payload["status"] == "PASSED"
    assert payload["execute_performed"] is True
    assert payload["job_count"] == 6
    state = json.loads((absurd / "jobs_state.json").read_text(encoding="utf-8"))
    assert len(state) == 6
    assert {row["state"] for row in state.values()} == {"QUEUED"}


def test_work_order_importer_next_actions_execute_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "work_orders.jsonl"
    actions = [
        {
            "action_id": "gap:abc123",
            "target_ref": "cluster-1",
            "plain_language_instruction": "Find direct evidence.",
            "closure_gate": "Attach source quote.",
        }
    ]
    proc = run([
        sys.executable,
        "scripts/work_order_importer.py",
        "next-actions",
        "--actions-json",
        json.dumps(actions),
        "--jsonl-out",
        str(out),
        "--execute",
        "--json",
    ])
    assert proc.returncode == 0, proc.stderr
    payload = json_line(proc.stdout)
    assert payload["status"] == "PASSED"
    assert payload["work_order_count"] == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["work_order_id"] == "wo:abc123"
    assert rows[0]["kind"] == "missing_evidence"
    assert rows[0]["closure_gate"] == "Attach source quote."
