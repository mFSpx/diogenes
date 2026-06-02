#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> dict:
    proc = subprocess.run([str(ROOT / "luci")] + args, cwd=ROOT, text=True, capture_output=True, check=True)
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert proc.stderr == "" or "WARNING:" in proc.stderr
    return json.loads(proc.stdout)


def test_learning_slice_visible_response_exposes_db_ids():
    payload = _run(
        [
            "learn",
            "--candidate-kind",
            "source",
            "--text",
            "study source candidate and prove it",
            "--artifact",
            "scripts/dev_journey_decision_points.py",
            "--json",
            "--run-id",
            "pytest-learn-visible-ids",
        ]
    )

    assert payload["learning_loop"]["work_order_id"] == payload["db_write"]["work_order_uuid"]
    assert payload["learning_loop"]["work_receipt_id"] == payload["db_write"]["work_receipt_uuid"]
    assert payload["learning_loop"]["raw_artifact_id"] == payload["db_write"]["raw_artifact_uuid"]
    assert payload["visible_response"]["work_order_id"] == payload["db_write"]["work_order_uuid"]
    assert payload["visible_response"]["work_receipt_id"] == payload["db_write"]["work_receipt_uuid"]
    assert payload["visible_response"]["attempt_id"] == payload["db_write"]["work_order_uuid"]
    assert payload["visible_response"]["raw_artifact_id"] == payload["db_write"]["raw_artifact_uuid"]


def test_source_slice_visible_response_exposes_db_ids():
    payload = _run(
        [
            "source",
            "--text",
            "read the live world from Hacker News and arXiv",
            "--json",
            "--run-id",
            "pytest-source-visible-ids",
        ]
    )

    assert payload["visible_response"]["work_order_id"] == payload["db_write"]["work_order_uuid"]
    assert payload["visible_response"]["work_receipt_id"] == payload["db_write"]["work_receipt_uuid"]
    assert payload["visible_response"]["attempt_id"] == payload["db_write"]["work_order_uuid"]
    assert payload["visible_response"]["raw_artifact_id"] == payload["db_write"]["raw_artifact_uuid"]
