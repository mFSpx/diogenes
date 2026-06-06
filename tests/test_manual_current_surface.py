from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_manual_current_exposes_live_operator_surface_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        payload = json.loads(body)

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert len(body) < 250_000, len(body)
    assert len(json.dumps(row["live_surface"], separators=(",", ":"))) < 200_000
    assert row["manual_id"] == "LUCIDOTA_OPERATOR_MANUAL"
    assert row["title"] == "LUCIDOTA Operator Manual"
    assert isinstance(row.get("route_list"), list) and row["route_list"], row
    assert isinstance(row.get("route_refs"), list) and row["route_refs"]
    assert isinstance(row.get("capability_refs"), list) and row["capability_refs"]
    assert isinstance(row.get("surface_refs"), list) and row["surface_refs"]
    assert isinstance(row.get("renderer_refs"), list) and row["renderer_refs"]
    assert isinstance(row.get("auth_expectations"), dict)
    assert row["auth_expectations"]["skill_layers"].startswith("execution aids only")
    assert isinstance(row.get("work_order_flow"), dict)
    assert isinstance(row.get("live_surface"), dict)
    assert isinstance(row.get("sub_orchestrators"), list)
    assert row["sub_orchestrators"], row
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert isinstance(row.get("blockers"), dict)
    assert isinstance(row.get("receipts"), dict)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert "cli_process_receipts" in row["receipts"]
    assert "flow_receipts" in row["receipts"]
    assert "api_test_execution_receipts" in row["receipts"]
    assert isinstance(row.get("next_commands"), list)
    assert isinstance(row.get("retired_surfaces"), list)
    assert "BOOKS folder watcher authority" in row["retired_surfaces"]
    routes = {route["route_id"] for route in row["route_list"]}
    assert {
        "manual_current",
        "indy_queue",
        "indy_responses",
        "daemon_status",
        "cloud_packet",
        "cli_process_receipts",
        "payload_archive_status",
        "skill_policy_current",
        "root_orchestrator_current",
        "chrono_current",
        "api_root_law_docs",
        "api_bible_subtree",
        "api_bible_edges",
        "api_bible_manuals",
        "api_bible_nodes",
        "api_bible_route_catalog",
        "api_workflow_registry",
        "agent_thread_runtime",
    }.issubset(routes)
    assert "manual_current" in row["route_refs"]
    assert "capability_current" in row["route_refs"]
    assert "root_orchestrator_current" in row["route_refs"]
    assert "daemon_status" in row["route_refs"]
    assert "cli_process_receipts" in row["route_refs"]
    assert "manual_current" in row["surface_refs"]
    assert {
        "api_bible_edges",
        "api_bible_manuals",
        "api_bible_nodes",
        "api_bible_route_catalog",
        "api_bible_subtree",
        "api_root_law_docs",
        "cloud_packet",
        "decompose_prompt_to_work_orders",
        "file_prompt",
        "link_prompt_work_order",
    }.issubset(routes)
    assert "skill_policy_current" in row["live_surface"]
    assert "root_orchestrator_current" in row["live_surface"]
    assert "chrono_current" in row["live_surface"]
    assert "daemon_status" in row["live_surface"]
    assert isinstance(row["live_surface"]["daemon_status"], list)
    assert row["live_surface"]["daemon_status"]
    daemon_row = row["live_surface"]["daemon_status"][0]
    assert "goal" in daemon_row
    assert "db_law" in daemon_row
    assert "next_commands" in daemon_row
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert all(isinstance(cmd, str) for cmd in row["next_commands"])
    assert not any(cmd.startswith("./luci") for cmd in row["next_commands"])
    assert "manual_current" in row["next_commands"]
    assert "root_orchestrator_current" in row["next_commands"]
    assert "daemon_status" in row["next_commands"]
    assert "indy_queue" in row["next_commands"]
    assert "indy_responses" in row["next_commands"]
    assert "capability_current" in row["next_command_refs"]
    assert "model_registry" in row["next_command_refs"]
    assert "provider_registry" in row["next_command_refs"]
    assert "workflow_registry" in row["next_command_refs"]
    assert "skill_policy_current" in row["next_command_refs"]
    assert "sub_orchestrator_threads" in row["next_command_refs"]
    assert "sub_orchestrator_grants" in row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "capability_current" in row["next_command_refs"]
    assert "command_registry" in row["live_surface"]
    assert isinstance(row["live_surface"].get("sub_orchestrator_threads"), list)
    assert row["live_surface"]["sub_orchestrator_threads"]
    assert isinstance(row["live_surface"].get("sub_orchestrator_grants"), list)
    assert row["live_surface"]["sub_orchestrator_grants"]
    assert isinstance(row["live_surface"].get("orchestration"), dict)
    assert row["live_surface"]["orchestration"]["mode"] == "sub_orchestrator"
    assert row["live_surface"]["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"
    assert "schema_owner_manifest" in row["live_surface"]
    assert "surface_registry" in row["live_surface"]
    assert "renderer_registry" in row["live_surface"]
    assert "controller_grant" in row["live_surface"]
    assert "agent_thread_runtime" in row["live_surface"]
    assert "root_law_docs" in row["live_surface"]
    assert "api_root_law_docs" in row["live_surface"]
    assert "api_route_catalog" in row["live_surface"]
    assert "api_test_execution_receipts" in row["live_surface"]
    assert "canon_current" in row["live_surface"]
    assert "cli_process_receipts" in row["live_surface"]
    assert "api_cli_process_receipts" in row["live_surface"]
    assert "api_flow_receipts" in row["live_surface"]
    assert "api_bytewax_windows" in row["live_surface"]
    assert "flow_specs" in row["live_surface"]
    assert "api_model_registry_current" in row["live_surface"]
    assert "api_provider_current" in row["live_surface"]
    assert "api_workflow_current" in row["live_surface"]
    assert "api_capability_current" in row["live_surface"]
    assert "api_sheet_current" in row["live_surface"]
    assert "api_model_routing_current" in row["live_surface"]
    assert "api_model_routing_blockers" in row["live_surface"]
    assert "api_model_registry_raw" in row["live_surface"]
    assert "queue_loop" in row["work_order_flow"]
    assert "api_route_catalog" in row["live_surface"]
    assert "sub-orchestrators packet" in row["auth_expectations"]["manual_source"].lower()
    assert "receipts packet" in row["auth_expectations"]["manual_source"].lower()
    assert "blocker packet" in row["auth_expectations"]["manual_source"].lower()
    assert "rpc alias packets" in row["auth_expectations"]["manual_source"].lower()
    assert "queue" in row["auth_expectations"]["manual_source"].lower()
    assert "bytewax" in row["auth_expectations"]["manual_source"].lower()
    assert "cloud_packet" in row["next_command_refs"]
    assert "decompose_prompt_to_work_orders" in row["next_command_refs"]
    assert "file_prompt" in row["next_command_refs"]
    assert "link_prompt_work_order" in row["next_command_refs"]
    assert not any("127.0.0.1:3000" in cmd for cmd in row["next_commands"])
