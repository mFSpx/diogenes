from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def _get_json(path: str):
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/{path}?limit=1", timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_orchestrator_registry_routes_are_exposed_over_postgrest() -> None:
    routes = [
        "manual_current",
        "canon_current",
        "canon_versions",
        "active_goal",
        "api_route_catalog",
        "api_workflow_registry",
        "ontology_work_batch",
        "ontology_work_item",
        "todo_current",
        "flow_specs",
        "capability_registry",
        "model_registry",
        "model_registry_current",
        "model_routing_current",
        "provider_registry",
        "provider_current",
        "workflow_registry",
        "capability_current",
        "workflow_current",
        "sheet_current",
        "daemon_status",
        "bytewax_compact_windows",
        "indy_queue",
        "indy_responses",
        "book_source",
        "book_scan",
        "book_read_queue",
        "book_note",
        "lora_candidate",
        "lora_adapter",
        "training_job",
        "book_receipt",
        "cli_process_receipts",
        "payload_archive_status",
        "skill_policy_current",
        "root_orchestrator_current",
        "chrono_current",
    ]

    for route in routes:
        status, payload = _get_json(route)
        assert status == 200, route
        assert isinstance(payload, list), route

    _, model_rows = _get_json("model_registry")
    if model_rows:
        assert "model_id" in model_rows[0]

    _, provider_rows = _get_json("provider_registry")
    assert provider_rows and provider_rows[0]["provider_key"] in {"codex", "vibe", "groq"}

    with urllib.request.urlopen(f"{LIVE_BASE_URL}/workflow_registry?limit=100", timeout=5) as resp:
        assert resp.status == 200
        workflow_rows = json.loads(resp.read().decode("utf-8"))
    workflow_names = {row["workflow_name"] for row in workflow_rows}
    assert {"basic-workflows", "indy-journal-wiki", "indy-daily-backup"}.issubset(workflow_names)

    _, daemon_rows = _get_json("daemon_status")
    assert isinstance(daemon_rows, list)

    _, manual_rows = _get_json("manual_current")
    assert manual_rows and manual_rows[0]["manual_id"] == "LUCIDOTA_OPERATOR_MANUAL"
    assert "route_list" in manual_rows[0]
    assert "next_commands" in manual_rows[0]
    route_ids = {route["route_id"] for route in manual_rows[0]["route_list"]}
    assert {"ontology_work_batch", "ontology_work_item", "todo_current", "skill_policy_current", "root_orchestrator_current", "chrono_current", "sheet_current", "cli_process_receipts", "payload_archive_status"}.issubset(route_ids)
    assert "skill_policy_current" in manual_rows[0]["live_surface"]
    assert "chrono_current" in manual_rows[0]["live_surface"]
    assert "canon_current" in manual_rows[0]["live_surface"]
    assert "model_registry_current" in manual_rows[0]["live_surface"]
    assert "model_routing_current" in manual_rows[0]["live_surface"]
    assert "capability_current" in manual_rows[0]["live_surface"]
    assert "provider_current" in manual_rows[0]["live_surface"]
    assert "workflow_current" in manual_rows[0]["live_surface"]
    assert "sheet_current" in manual_rows[0]["live_surface"]
