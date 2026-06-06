from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_canon_current_reports_live_canon_node() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/canon_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["manual_id"] == "FLIGHT_MAN"
    assert row["node_id"]
    assert row["title"]
    assert row["status"]
    assert isinstance(row.get("ontology_tags"), list)
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"


def test_manual_current_mentions_canon_route() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "canon_current" in route_ids
    assert "canon_current" in row["live_surface"]
    assert "canon" in row["auth_expectations"]["manual_source"]
