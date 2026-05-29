#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_absurd_chrono_worker_rejects_non_chrono_queue_without_db_write():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/absurd_chrono_worker.py"),
            "--action",
            "audit",
            "--queue",
            "wrong",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    report_line = next(line for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    report_path = ROOT / report_line.split("=", 1)[1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["requested_queue"] == "wrong"
    assert report["expected_queue"] == "chrono"
    assert report["db_writes_performed"] is False
    assert report["temporal_claims_mutated_by_wrapper"] is False
    assert report["canonical_graph_writes_performed"] is False
    assert report["blockers"] == ["queue_mismatch:wrong!=chrono"]
