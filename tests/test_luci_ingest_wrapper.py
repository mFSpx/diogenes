#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_markdown_ingest_class():
    proc = subprocess.run(
        [str(ROOT / "luci"), "ingest", "markdown", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["count"] > 0
    assert payload["graph_approval_mode"] == "staged"
    assert payload["executed"] is False
    receipt = ROOT / payload["receipt_path"]
    assert receipt.exists()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["schema"] == "lucidota.markdown_ingest_archive.receipt.v1"
    assert receipt_payload["count"] == payload["count"]
    assert receipt_payload["receipt_path"] == payload["receipt_path"]


def test_luci_shell_wrapper_markdown_ingest_receipt_path_is_stable_for_same_repo_state():
    first = subprocess.run(
        [str(ROOT / "luci"), "ingest", "markdown", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    second = subprocess.run(
        [str(ROOT / "luci"), "ingest", "markdown", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload_a = json.loads(first.stdout)
    payload_b = json.loads(second.stdout)
    assert payload_a["receipt_path"] == payload_b["receipt_path"]
    assert payload_a["count"] == payload_b["count"]
