from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_sheet_current_reports_spreadsheet_status_and_next_batch() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/sheet_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["sheet_packet_id"] == "sheet_current"
    assert isinstance(row.get("sheet_tasks"), dict)
    assert isinstance(row.get("projections"), dict)
    assert isinstance(row.get("active_work"), list)
    assert isinstance(row.get("next_work_batch"), list)
    assert isinstance(row.get("case_pressure_sheet"), list)
    assert row["sheet_tasks"]["sheet_task_count"] >= 0
    assert "routing_order" in row["sheet_notes"]
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert isinstance(row.get("next_commands"), list) and row["next_commands"]
    assert row["next_commands"] == ["sheet_current"]
    assert "sheet_current" in row["next_command_refs"]
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert row["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"
    assert row["goal"]["title"] == "Spreadsheet current packet"


def test_manual_current_mentions_sheet_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "sheet_current" in route_ids
    assert "sheet_current" in row["live_surface"]
    assert "sheet current packet" in row["auth_expectations"]["manual_source"]
    assert isinstance(row["live_surface"]["sheet_current"], list)
    assert row["live_surface"]["sheet_current"]
