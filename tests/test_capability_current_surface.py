from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_capability_current_reports_registry_summary_and_active_capabilities() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/capability_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["capability_packet_id"] == "capability_current"
    assert isinstance(row.get("capability_summary"), dict)
    assert isinstance(row.get("status_breakdown"), dict)
    assert isinstance(row.get("group_breakdown"), dict)
    assert isinstance(row.get("active_capabilities"), list)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert isinstance(row.get("next_commands"), list)
    assert "capability_current" in row["next_commands"]
    assert "capability_registry" in row["next_commands"]
    assert "workflow_current" in row["next_commands"]
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert row["capability_summary"]["capability_count"] >= row["capability_summary"]["active_count"]
    assert "workflow_names" in row["capability_summary"]
    assert "routing_notes" in row


def test_manual_current_mentions_capability_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "capability_current" in route_ids
    assert "capability_current" in row["live_surface"]
    assert "capability registry" in row["auth_expectations"]["manual_source"]
