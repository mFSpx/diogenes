from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_root_orchestrator_current_route_is_readable_and_manual_references_it() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/root_orchestrator_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["orchestrator_id"] == "ROOT_ORCHESTRATOR_CURRENT"
    assert row["title"] == "Root Orchestrator"
    assert isinstance(row.get("live_surface"), dict)
    assert isinstance(row.get("route_list"), list)
    assert len(json.dumps(row["live_surface"], ensure_ascii=False)) < 1_000_000
    assert row.get("route_count", 0) > 0
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert isinstance(row.get("next_commands"), list) and row["next_commands"]
    assert isinstance(row.get("sub_orchestrators"), list) and row["sub_orchestrators"], row
    assert isinstance(row.get("sub_orchestrator_threads"), list) and row["sub_orchestrator_threads"], row
    assert isinstance(row.get("sub_orchestrator_grants"), list) and row["sub_orchestrator_grants"], row
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert isinstance(row.get("auth_expectations"), dict)
    assert isinstance(row.get("work_order_flow"), dict)
    assert isinstance(row.get("blockers"), dict)
    assert isinstance(row.get("receipts"), dict)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "capability_current" in row["next_command_refs"]
    assert "capability_registry" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert "model_registry" in row["next_command_refs"]
    assert "provider_registry" in row["next_command_refs"]
    assert "workflow_registry" in row["next_command_refs"]
    assert "skill_policy_current" in row["next_command_refs"]
    assert "command_registry" in row["live_surface"]
    assert "api_route_catalog" in row["next_command_refs"]
    assert "flow_specs" in row["next_command_refs"]
    assert "todo_current" in row["next_command_refs"]
    assert "sub_orchestrator_threads" in row["next_command_refs"]
    assert "sub_orchestrator_grants" in row["next_command_refs"]
    assert "cli_process_receipts" in row["next_command_refs"]
    assert all(isinstance(cmd, str) for cmd in row["next_commands"])
    assert not any(cmd.startswith("./luci") for cmd in row["next_commands"])
    assert set(row["next_commands"]).issubset(set(row["next_command_refs"]))
    assert "manual_current" in row["next_commands"]
    assert "root_orchestrator_current" in row["next_commands"]
    assert "daemon_status" in row["next_commands"]
    assert "indy_queue" in row["next_commands"]
    assert "indy_responses" in row["next_commands"]
    assert "api_route_catalog" in row["next_commands"]
    assert not any("127.0.0.1:3000" in cmd for cmd in row["next_commands"])
    assert row["sub_orchestrator_threads"][0]["thread_key"] == "root_operator_thread"
    assert any(thread["thread_key"] == "sub_orchestrator_thread" for thread in row["sub_orchestrator_threads"])
    assert row["sub_orchestrator_threads"][1]["controller_grant_key"] == row["sub_orchestrator_grants"][0]["grant_key"]
    assert row["sub_orchestrator_grants"][0]["grant_key"] == "default_local_operator"
    assert row["sub_orchestrator_grants"][0]["effective_status"] == "active"
    assert "cloud_packet" in row["next_command_refs"]
    assert "decompose_prompt_to_work_orders" in row["next_command_refs"]
    assert "file_prompt" in row["next_command_refs"]
    assert "link_prompt_work_order" in row["next_command_refs"]
    assert not any("127.0.0.1:3000" in cmd for cmd in row["next_commands"])
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "manual_current" in route_ids
    assert "capability_current" in route_ids
    assert "ontology_work_batch" in route_ids
    assert "ontology_work_item" in route_ids
    assert "prompt_recent" in route_ids
    assert "prompt_catalog_status" in route_ids
    assert "prompts_filed" in route_ids
    assert "prompt_work_order_links" in route_ids
    assert "prompt_unlinked" in route_ids
    assert "cli_process_receipts" in route_ids
    assert "flow_receipts" in route_ids
    assert "api_test_execution_receipts" in route_ids
    assert "flow_specs" in route_ids
    assert "payload_archive_status" in route_ids
    assert "bytewax_compact_windows" in route_ids
    assert "indy_queue" in route_ids
    assert "indy_responses" in route_ids
    assert "cloud_packet" in route_ids
    assert "root_law_docs" in route_ids
    assert "api_root_law_docs" in route_ids
    assert "api_bible_edges" in route_ids
    assert "api_bible_manuals" in route_ids
    assert "api_bible_nodes" in route_ids
    assert "api_bible_route_catalog" in route_ids
    assert "api_bible_subtree" in route_ids
    assert "api_workflow_registry" in route_ids
    assert "fn_bible_node_sort_key" in route_ids
    assert "get_subtree" in route_ids
    assert isinstance(row["live_surface"].get("sub_orchestrators"), list)
    assert row["live_surface"]["sub_orchestrators"]
    assert row["live_surface"]["sub_orchestrators"][0]["route_id"]
    assert isinstance(row["live_surface"].get("orchestration"), dict)
    assert row["live_surface"]["orchestration"]["mode"] == "sub_orchestrator"
    assert row["live_surface"]["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"
    assert "daemon_status" in row["live_surface"]
    assert isinstance(row["live_surface"]["daemon_status"], list)
    assert row["live_surface"]["daemon_status"]
    daemon_row = row["live_surface"]["daemon_status"][0]
    assert "goal" in daemon_row
    assert "db_law" in daemon_row
    assert "next_commands" in daemon_row
    assert "model_registry" in row["live_surface"]
    assert "provider_registry" in row["live_surface"]
    assert "workflow_registry" in row["live_surface"]
    assert "chrono_current" in row["live_surface"]
    assert "capability_registry" in row["live_surface"]
    assert "model_registry_current" in row["live_surface"]
    assert "provider_current" in row["live_surface"]
    assert "workflow_current" in row["live_surface"]
    assert "capability_current" in row["live_surface"]
    assert "sheet_current" in row["live_surface"]
    assert isinstance(row["live_surface"]["sheet_current"], list)
    assert row["live_surface"]["sheet_current"]
    sheet_row = row["live_surface"]["sheet_current"][0]
    assert "goal" in sheet_row
    assert "db_law" in sheet_row
    assert "next_commands" in sheet_row
    assert "model_routing_current" in row["live_surface"]
    assert "model_routing_blockers" in row["live_surface"]
    assert isinstance(row["live_surface"]["model_routing_blockers"], list)
    assert row["live_surface"]["model_routing_blockers"]
    blocker_row = row["live_surface"]["model_routing_blockers"][0]
    assert "goal" in blocker_row
    assert "db_law" in blocker_row
    assert "next_commands" in blocker_row
    assert "indy_queue" in row["live_surface"]
    assert "indy_responses" in row["live_surface"]
    assert "cli_process_receipts" in row["live_surface"]
    assert "flow_receipts" in row["live_surface"]
    assert "selected_lanes" in row["live_surface"]
    assert "missing_executor_roles" in row["live_surface"]
    assert "fn_bible_node_sort_key" in row["live_surface"]
    assert "get_subtree" in row["live_surface"]
    assert "todo_current" in row["live_surface"]
    assert "current_goal" in row["live_surface"]
    assert "prompt_catalog_status" in row["live_surface"]
    assert "api_route_catalog" in row["live_surface"]
    assert "canon_current" in row["live_surface"]
    assert "canon_versions" in row["live_surface"]
    assert "skill_policy_current" in row["live_surface"]
    assert "todo_current" in row["live_surface"]
    assert isinstance(row["live_surface"]["todo_current"], list)
    assert row["live_surface"]["todo_current"]
    todo_row = row["live_surface"]["todo_current"][0]
    assert "goal" in todo_row
    assert "db_law" in todo_row
    assert "next_commands" in todo_row
    assert "api_bible_edges" in row["live_surface"]
    assert "api_bible_manuals" in row["live_surface"]
    assert "api_bible_nodes" in row["live_surface"]
    assert "api_bible_route_catalog" in row["live_surface"]
    assert "api_bible_subtree" in row["live_surface"]
    assert "api_root_law_docs" in row["live_surface"]
    assert "api_workflow_registry" in row["live_surface"]
    assert "api_capability_registry" in row["live_surface"]
    assert "api_provider_registry" in row["live_surface"]
    assert "api_model_registry_raw" in row["live_surface"]
    assert "api_bytewax_windows" in row["live_surface"]
    assert "api_cloud_packet" in row["live_surface"]
    assert "api_model_registry_current" in row["live_surface"]
    assert "api_provider_current" in row["live_surface"]
    assert "api_workflow_current" in row["live_surface"]
    assert "api_capability_current" in row["live_surface"]
    assert "api_sheet_current" in row["live_surface"]
    assert "api_model_routing_current" in row["live_surface"]
    assert "api_model_routing_blockers" in row["live_surface"]
    assert "book_source" in row["live_surface"]
    assert "book_scan" in row["live_surface"]
    assert "book_read_queue" in row["live_surface"]
    assert "book_note" in row["live_surface"]
    assert "book_receipt" in row["live_surface"]
    assert "lora_candidate" in row["live_surface"]
    assert "lora_adapter" in row["live_surface"]
    assert "training_job" in row["live_surface"]
    assert "ontology_work_batch" in row["live_surface"]
    assert "ontology_work_item" in row["live_surface"]
    assert "prompt_recent" in row["live_surface"]
    assert "prompts_filed" in row["live_surface"]
    assert "prompt_work_order_links" in row["live_surface"]
    assert "prompt_unlinked" in row["live_surface"]
    assert "flow_specs" in row["live_surface"]
    assert "payload_archive_status" in row["live_surface"]
    assert "bytewax_compact_windows" in row["live_surface"]
    assert "cloud_packet" in row["live_surface"]
    assert "blockers" in row["live_surface"]
    assert "receipts" in row["live_surface"]
    assert "chrono_current" in route_ids
    assert "capability_registry" in route_ids
    assert "manual_source" in row["auth_expectations"]
    assert "root_loop" in row["work_order_flow"]

    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        manual_rows = json.loads(resp.read().decode("utf-8"))

    assert manual_rows and manual_rows[0]["manual_id"] == "LUCIDOTA_OPERATOR_MANUAL"
    manual_row = manual_rows[0]
    route_ids = {route["route_id"] for route in manual_row["route_list"]}
    assert "root_orchestrator_current" in route_ids
    assert "root_orchestrator_current" in manual_row["live_surface"]
    assert manual_row["live_surface"]["root_orchestrator_current"][0]["orchestrator_id"] == "ROOT_ORCHESTRATOR_CURRENT"
