#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_operator_route_class(tmp_path):
    src = tmp_path / "drop"
    src.mkdir()
    (src / "note.md").write_text("Alice saw Evidence.")
    receipt_dir = tmp_path / "receipts"
    base_dir = tmp_path / "cases"

    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "operator-route",
            "--raw-command",
            "create case from folder and build packet",
            "--case-id",
            "operator-case",
            "--source-folder",
            str(src),
            "--base-dir",
            str(base_dir),
            "--receipt-dir",
            str(receipt_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "OPERATOR_ROUTE=PASSED" in proc.stdout
    assert "operator_command_router" in proc.stdout
    assert "receipt_path" in proc.stdout


def test_luci_shell_wrapper_exposes_operator_route_json_only(tmp_path):
    src = tmp_path / "drop"
    src.mkdir()
    (src / "note.md").write_text("Alice saw Evidence.")
    receipt_dir = tmp_path / "receipts"
    base_dir = tmp_path / "cases"

    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "operator-route",
            "--raw-command",
            "create case from folder and build packet",
            "--case-id",
            "operator-case-json",
            "--source-folder",
            str(src),
            "--base-dir",
            str(base_dir),
            "--receipt-dir",
            str(receipt_dir),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "OPERATOR_ROUTE=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
