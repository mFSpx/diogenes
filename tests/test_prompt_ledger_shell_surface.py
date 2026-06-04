#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_ledger_shell_alias_reads_live_recent_and_catalog() -> None:
    recent = subprocess.run(
        [str(ROOT / "luci"), "prompt", "recent", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    catalog = subprocess.run(
        [str(ROOT / "luci"), "prompt", "catalog", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    recent_payload = json.loads(recent.stdout)
    catalog_payload = json.loads(catalog.stdout)
    assert isinstance(recent_payload, list) and recent_payload
    assert isinstance(catalog_payload, list) and catalog_payload
    assert "prompt_id" in recent_payload[0]
    assert "prompt_count" in catalog_payload[0]
