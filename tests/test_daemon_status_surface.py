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
    assert "next_command_refs" in row
    assert isinstance(row["next_command_refs"], list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "capability_current" in row["next_command_refs"]
    assert "model_registry" in row["next_command_refs"]
    assert "provider_registry" in row["next_command_refs"]
    assert "workflow_registry" in row["next_command_refs"]
    assert "skill_policy_current" in row["next_command_refs"]
    assert "payload_archive_status" in row["next_command_refs"]
    assert "cli_process_receipts" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert row["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"
    assert "next_commands" in row
    assert isinstance(row["next_commands"], list) and row["next_commands"]
    assert row["next_commands"] == ["daemon_status"]
    assert "daemon_status" in row["next_command_refs"]


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
