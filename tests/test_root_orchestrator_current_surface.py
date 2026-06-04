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
    assert row.get("route_count", 0) > 0
    assert isinstance(row.get("next_commands"), list) and row["next_commands"]
    assert isinstance(row.get("sub_orchestrators"), list) and row["sub_orchestrators"], row
    assert isinstance(row.get("auth_expectations"), dict)
    assert isinstance(row.get("work_order_flow"), dict)
    assert isinstance(row.get("blockers"), dict)
    assert isinstance(row.get("receipts"), dict)
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert any("luci root orchestrator current --json" in cmd for cmd in row["next_commands"])
    assert any("luci openapi --json" in cmd for cmd in row["next_commands"])
    assert any("luci payload-archive-status --json" in cmd for cmd in row["next_commands"])
    assert any("luci model-routing-current --json" in cmd for cmd in row["next_commands"])
    assert any("luci model-routing-blockers --json" in cmd for cmd in row["next_commands"])
    assert any("luci model registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci model registry current --json" in cmd for cmd in row["next_commands"])
    assert any("luci model registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci provider current --json" in cmd for cmd in row["next_commands"])
    assert any("luci provider registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci provider registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci workflow current --json" in cmd for cmd in row["next_commands"])
    assert any("luci workflow registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci capability current --json" in cmd for cmd in row["next_commands"])
    assert any("luci capability registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci capability registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci sheet current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api manual current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api active goal --json" in cmd for cmd in row["next_commands"])
    assert any("luci api daemon status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api route catalog --json" in cmd for cmd in row["next_commands"])
    assert any("luci api test execution receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt catalog status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt filed --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt links --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt raw recent --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt raw filed --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt raw links --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt raw unlinked --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt raw catalog --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt recent --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompts filed --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt work-order links --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt unlinked --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt catalog --json" in cmd for cmd in row["next_commands"])
    assert any("luci api root law docs --json" in cmd for cmd in row["next_commands"])
    assert any("luci api root orchestrator current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api manual current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api chrono current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api payload archive status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api canon current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api canon versions --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible edges --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible manuals --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible nodes --manual-id RUNTIME_GOVERNOR --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible route catalog --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible subtree --root-id 1.0.0 --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model registry current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api provider current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api provider registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci api workflow current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api workflow registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api capability current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api capability registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci api sheet current --json" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/manual_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/root_orchestrator_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/canon_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/active_goal?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/capability_registry?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/model_registry?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/provider_registry?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/workflow_registry?limit=1" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/daemon_status?limit=5" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/indy_queue?limit=5" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/indy_responses?limit=5" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/bytewax_compact_windows?limit=5" in cmd for cmd in row["next_commands"])
    assert any("luci api model routing current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model routing blockers --json" in cmd for cmd in row["next_commands"])
    assert any("luci model routing current --json" in cmd for cmd in row["next_commands"])
    assert any("luci model routing blockers --json" in cmd for cmd in row["next_commands"])
    assert any("luci api indy queue --json" in cmd for cmd in row["next_commands"])
    assert any("luci api indy responses --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bytewax windows --json" in cmd for cmd in row["next_commands"])
    assert any("luci bytewax raw windows --json" in cmd for cmd in row["next_commands"])
    assert any("luci api cli process receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api flow receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api flow specs --json" in cmd for cmd in row["next_commands"])
    assert any("luci api skill policy current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api todo current --json" in cmd for cmd in row["next_commands"])
    assert any("luci root-law-docs --json" in cmd for cmd in row["next_commands"])
    assert any("luci api cloud packet --work-order-id 00000000-0000-0000-0000-000000000000 --json" in cmd for cmd in row["next_commands"])
    assert any("luci prompt recent --json" in cmd for cmd in row["next_commands"])
    assert any("luci prompt catalog status --json" in cmd for cmd in row["next_commands"])
    assert any("luci flow specs --json" in cmd for cmd in row["next_commands"])
    assert any("luci flow receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc cloud-packet --work-order-id 00000000-0000-0000-0000-000000000000 --json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc decompose-prompt --payload-json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc file-prompt --payload-json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc link-prompt --payload-json" in cmd for cmd in row["next_commands"])
    assert any("luci api book source --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book scan --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book read-queue --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book note --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book receipt --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book adapter --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book candidate --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book training --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw source --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw scan --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw read-queue --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw note --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw receipt --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw adapter --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw candidate --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw training --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book read queue --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bytewax compact windows --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bytewax raw windows --json" in cmd for cmd in row["next_commands"])
    assert any("scripts/indy_daemon.py --once --json" in cmd for cmd in row["next_commands"])
    assert any("scripts/indy_runtime_broker.py snapshot --json" in cmd for cmd in row["next_commands"])
    assert any("scripts/luci_todo.py --json" in cmd for cmd in row["next_commands"])
    assert any("scripts/ontology_work_compiler.py --json --text \"<objective text>\"" in cmd for cmd in row["next_commands"])
    assert any("scripts/ontology_work_compiler.py --json --text \"<operator objective>\"" in cmd for cmd in row["next_commands"])
    assert any("scripts/prompt_ledger_capture.py --json" in cmd for cmd in row["next_commands"])
    assert any("test_receipt_gate.py run --scope policy_and_retirement" in cmd for cmd in row["next_commands"])
    assert any("rpc/cloud_packet" in cmd and "include_raw_bodies" in cmd for cmd in row["next_commands"])
    assert any("rpc/decompose_prompt_to_work_orders" in cmd for cmd in row["next_commands"])
    assert any("rpc/file_prompt" in cmd for cmd in row["next_commands"])
    assert any("rpc/link_prompt_work_order" in cmd for cmd in row["next_commands"])
    assert any("curl -sS http://127.0.0.1:3000/" in cmd for cmd in row["next_commands"])
    assert any("api_bible_edges?limit=5" in cmd for cmd in row["next_commands"])
    assert any("api_bible_manuals?limit=5" in cmd for cmd in row["next_commands"])
    assert any("api_bible_nodes?manual_id=eq.RUNTIME_GOVERNOR" in cmd for cmd in row["next_commands"])
    assert any("api_bible_route_catalog?limit=5" in cmd for cmd in row["next_commands"])
    assert any("api_bible_subtree?root_id=eq.1.0.0" in cmd for cmd in row["next_commands"])
    assert any("api_root_law_docs?limit=1" in cmd for cmd in row["next_commands"])
    assert any("api_route_catalog?limit=1" in cmd for cmd in row["next_commands"])
    assert any("api_test_execution_receipts?limit=1" in cmd for cmd in row["next_commands"])
    assert any("api_test_execution_receipts?limit=3" in cmd for cmd in row["next_commands"])
    assert any("canon_versions?limit=5" in cmd for cmd in row["next_commands"])
    assert any("capability_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("chrono_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("cli_process_receipts?limit=3" in cmd for cmd in row["next_commands"])
    assert any("flow_receipts?limit=1" in cmd for cmd in row["next_commands"])
    assert any("flow_receipts?limit=3" in cmd for cmd in row["next_commands"])
    assert any("flow_specs?limit=1" in cmd for cmd in row["next_commands"])
    assert any("model_registry?limit=20" in cmd for cmd in row["next_commands"])
    assert any("model_registry_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("model_routing_blockers?limit=1" in cmd for cmd in row["next_commands"])
    assert any("model_routing_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("payload_archive_status?limit=6" in cmd for cmd in row["next_commands"])
    assert any("prompt_catalog_status?limit=1" in cmd for cmd in row["next_commands"])
    assert any("prompt_recent?limit=5" in cmd for cmd in row["next_commands"])
    assert any("provider_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("sheet_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("skill_policy_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("todo_current?limit=5" in cmd for cmd in row["next_commands"])
    assert any("workflow_current?limit=1" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work batch --json" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work item --json" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work raw batch --json" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work raw item --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt recent --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt catalog status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api root-law-docs --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api provider registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci api provider registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api workflow registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api capability registry raw --json" in cmd for cmd in row["next_commands"])
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

    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        manual_rows = json.loads(resp.read().decode("utf-8"))

    assert manual_rows and manual_rows[0]["manual_id"] == "LUCIDOTA_OPERATOR_MANUAL"
    manual_row = manual_rows[0]
    route_ids = {route["route_id"] for route in manual_row["route_list"]}
    assert "root_orchestrator_current" in route_ids
    assert "root_orchestrator_current" in manual_row["live_surface"]
    assert manual_row["live_surface"]["root_orchestrator_current"][0]["orchestrator_id"] == "ROOT_ORCHESTRATOR_CURRENT"
