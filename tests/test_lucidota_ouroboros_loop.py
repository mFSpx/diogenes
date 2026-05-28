#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lucidota_ouroboros_loop import classify_path, discover_targets, run_loops, validation_command


def test_classify_path_respects_active_supporting_and_receipts():
    assert classify_path(ROOT / "scripts/lucidota_ouroboros_loop.py") == "ACTIVE"
    assert classify_path(ROOT / "scripts/legacy/boring_beast.py") == "LEGACY_USEFUL"
    assert classify_path(ROOT / "05_OUTPUTS/example_receipt.json") == "RECEIPT"
    assert classify_path(ROOT / "scripts/some_helper.py") == "SUPPORTING"


def test_validation_command_uses_smallest_safe_checker():
    assert validation_command(ROOT / "scripts/lucidota_ouroboros_loop.py")[:3] == [sys.executable, "-m", "py_compile"]
    assert validation_command(ROOT / "scripts/lucidota_synthesis_pass.sh") == ["bash", "-n", "scripts/lucidota_synthesis_pass.sh"]
    assert validation_command(ROOT / "06_SCHEMA/001_extensions.sql") is None


def test_run_loops_writes_summary_and_jsonl(tmp_path):
    summary = run_loops(
        loops=3,
        targets=[ROOT / "scripts/lucidota_ouroboros_loop.py", ROOT / "scripts/lucidota_synthesis_pass.sh"],
        receipt_root=tmp_path,
        timeout_sec=30,
    )
    assert summary["status"] == "PASS"
    assert summary["loops_executed"] == 3
    ledger = ROOT / summary["ledger_path"] if not Path(summary["ledger_path"]).is_absolute() else Path(summary["ledger_path"])
    if not ledger.exists():
        ledger = tmp_path / Path(summary["ledger_path"]).name
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["schema"] == "lucidota.ouroboros_loop.cycle.v1"
    assert rows[0]["target"]["classification"] == "ACTIVE"


def test_ouroboros_cli_smoke(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/lucidota_ouroboros_loop.py"),
            "--loops",
            "2",
            "--receipt-root",
            str(tmp_path),
            "--target",
            "scripts/lucidota_ouroboros_loop.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "REPORT_PATH=" in proc.stdout
    assert "LEDGER_PATH=" in proc.stdout
    assert "OUROBOROS_LOOP=PASS" in proc.stdout


def test_discover_targets_prioritizes_active_files():
    targets = [str(p.relative_to(ROOT)) for p in discover_targets(limit=8)]
    assert "scripts/lucidota_ouroboros_loop.py" in targets
    assert "scripts/lucidota_synthesis_pass.py" in targets
