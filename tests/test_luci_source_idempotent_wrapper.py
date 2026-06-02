#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_source_can_rerun_same_run_id_without_duplicate_raw_ref():
    run_id = "pytest-source-idempotent"
    args = [
        str(ROOT / "luci"),
        "source",
        "--text",
        "read the live world from Hacker News and arXiv",
        "--source",
        "arxiv",
        "--run-id",
        run_id,
        "--json",
    ]

    first = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True)
    second = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True)

    assert '"status": "DEGRADED"' in first.stdout or '"status":"DEGRADED"' in first.stdout or '"status": "PASS"' in first.stdout
    assert "db_error" not in second.stdout
    assert "UniqueViolation" not in second.stdout
    assert second.stdout.lstrip().startswith("{")
    assert second.stdout.rstrip().endswith("}")
    assert "RECEIPT_PATH=" not in second.stdout
    assert "SOURCE=" not in second.stdout


def test_luci_source_json_stdout_is_pure_json():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "source",
            "--text",
            "read the live world from Hacker News and arXiv",
            "--source",
            "arxiv",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "RECEIPT_PATH=" not in proc.stdout
    assert "SOURCE=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
