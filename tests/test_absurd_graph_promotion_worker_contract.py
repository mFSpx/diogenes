#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report_from(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def test_graph_promotion_enqueue_blocks_invalid_payload_json_without_db():
    proc = subprocess.run(
        [sys.executable, "scripts/absurd_graph_promotion_worker.py", "enqueue", "--payload-json", "not-json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2
    report = report_from(proc.stdout)
    assert report["status"] == "BLOCKED"
    assert report["execute_performed"] is False
    assert "Expecting value" in report["error"]


def test_graph_promotion_enqueue_blocks_non_object_payload_without_db():
    proc = subprocess.run(
        [sys.executable, "scripts/absurd_graph_promotion_worker.py", "enqueue", "--payload-json", "[]"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2
    report = report_from(proc.stdout)
    assert report["status"] == "BLOCKED"
    assert report["error"] == "payload_json_must_be_object"
