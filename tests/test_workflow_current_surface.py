from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_workflow_current_reports_registry_status_and_basic_workflows() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/workflow_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["workflow_packet_id"] == "workflow_current"
    assert isinstance(row.get("workflow_summary"), dict)
    assert isinstance(row.get("status_breakdown"), dict)
    assert isinstance(row.get("active_workflows"), list)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert any("luci workflow current --json" in cmd for cmd in row["next_commands"])
    assert row["workflow_summary"]["workflow_count"] >= 1
    assert row["workflow_summary"]["active_count"] >= 1
    assert "basic-workflows" in row["workflow_summary"]["active_names"]
    assert "next_action" in row["workflow_notes"]


def test_manual_current_mentions_workflow_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "workflow_current" in route_ids
    assert "workflow_current" in row["live_surface"]
    assert "workflow current packet" in row["auth_expectations"]["manual_source"]
