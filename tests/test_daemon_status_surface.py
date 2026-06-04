#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import urllib.request

LIVE_BASE_URL = "http://127.0.0.1:3000"
ROOT = Path(__file__).resolve().parents[1]


def test_daemon_status_route_is_readable() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/daemon_status?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "daemon_name" in row
    assert "heartbeat_kind" in row
    assert "process_id" in row
    assert "goal" in row
    assert "db_law" in row
    assert "next_commands" in row
    assert isinstance(row["next_commands"], list) and row["next_commands"]


def test_luci_daemon_status_text_renderer_shows_truth_spine() -> None:
    proc = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "luci_daemon_status.py"), "--limit", "1"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "DAEMON STATUS" in proc.stdout
    assert "goal=" in proc.stdout
    assert "db_law=" in proc.stdout
    assert "next_commands=" in proc.stdout
