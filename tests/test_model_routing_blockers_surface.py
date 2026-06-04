from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path


LIVE_BASE_URL = "http://127.0.0.1:3000"
ROOT = Path(__file__).resolve().parents[1]


def test_model_routing_blockers_reports_missing_roles_as_live_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/model_routing_blockers?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["routing_packet_id"] == "model_routing_blockers"
    assert isinstance(row.get("missing_roles"), list)
    assert row["missing_role_count"] == len(row["missing_roles"])
    assert "classifier" in row["missing_roles"]
    assert "treelite_gate" in row["missing_roles"]
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert isinstance(row.get("next_commands"), list) and row["next_commands"]


def test_manual_current_mentions_model_routing_blockers_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "model_routing_blockers" in route_ids
    assert "model_routing_blockers" in row["live_surface"]
    assert any("luci model-routing-blockers --json" in cmd for cmd in row["next_commands"])
    assert "model routing blockers" in row["auth_expectations"]["manual_source"].lower()
    assert isinstance(row["live_surface"]["model_routing_blockers"], list)
    assert row["live_surface"]["model_routing_blockers"]


def test_model_routing_blockers_spaced_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "model", "routing", "blockers", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/model_routing_blockers?limit=1")
    assert payload["payload"]
