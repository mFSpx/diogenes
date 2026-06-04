from __future__ import annotations

import json
import urllib.request


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_manual_current_exposes_live_operator_surface_packet() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["manual_id"] == "LUCIDOTA_OPERATOR_MANUAL"
    assert row["title"] == "LUCIDOTA Operator Manual"
    assert isinstance(row.get("route_list"), list) and row["route_list"], row
    assert isinstance(row.get("auth_expectations"), dict)
    assert row["auth_expectations"]["skill_layers"].startswith("execution aids only")
    assert isinstance(row.get("work_order_flow"), dict)
    assert isinstance(row.get("live_surface"), dict)
    assert isinstance(row.get("sub_orchestrators"), list)
    assert row["sub_orchestrators"], row
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
        "root_law_docs",
        "subtree",
    }.issubset(routes)
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
        "fn_bible_node_material",
        "fn_bible_node_sort_key",
        "get_subtree",
        "link_prompt_work_order",
    }.issubset(routes)
    assert {
        "rpc/cloud_packet",
        "rpc/decompose_prompt_to_work_orders",
        "rpc/file_prompt",
        "rpc/fn_bible_node_material",
        "rpc/fn_bible_node_sort_key",
        "rpc/get_subtree",
        "rpc/link_prompt_work_order",
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
    assert "sheet_current" in row["live_surface"]
    assert isinstance(row["live_surface"]["sheet_current"], list)
    assert row["live_surface"]["sheet_current"]
    sheet_row = row["live_surface"]["sheet_current"][0]
    assert "goal" in sheet_row
    assert "db_law" in sheet_row
    assert "next_commands" in sheet_row
    assert "model_routing_blockers" in row["live_surface"]
    assert isinstance(row["live_surface"]["model_routing_blockers"], list)
    assert row["live_surface"]["model_routing_blockers"]
    blocker_row = row["live_surface"]["model_routing_blockers"][0]
    assert "goal" in blocker_row
    assert "db_law" in blocker_row
    assert "next_commands" in blocker_row
    assert "api_capability_registry" in row["live_surface"]
    assert "api_provider_registry" in row["live_surface"]
    assert "api_workflow_registry" in row["live_surface"]
    assert "api_cloud_packet" in row["live_surface"]
    assert "capability_registry" in row["live_surface"]
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
    assert "api_bible_subtree" in row["live_surface"]
    assert "canon_versions" in row["live_surface"]
    assert "indy_queue" in row["live_surface"]
    assert "indy_responses" in row["live_surface"]
    assert "bytewax_compact_windows" in row["live_surface"]
    assert "cloud_packet" in row["live_surface"]
    assert "queue_loop" in row["work_order_flow"]
    assert "api_route_catalog" in row["live_surface"]
    assert "prompt_recent" in row["live_surface"]
    assert "prompts_filed" in row["live_surface"]
    assert "prompt_work_order_links" in row["live_surface"]
    assert "prompt_unlinked" in row["live_surface"]
    assert "prompt_catalog_status" in row["live_surface"]
    assert "todo_current" in row["live_surface"]
    assert isinstance(row["live_surface"]["todo_current"], list)
    assert row["live_surface"]["todo_current"]
    todo_row = row["live_surface"]["todo_current"][0]
    assert "goal" in todo_row
    assert "db_law" in todo_row
    assert "next_commands" in todo_row
    assert "missing_executor_roles" in row["live_surface"]
    assert "selected_lanes" in row["live_surface"]
    assert "sub-orchestrators packet" in row["auth_expectations"]["manual_source"].lower()
    assert "receipts packet" in row["auth_expectations"]["manual_source"].lower()
    assert "blocker packet" in row["auth_expectations"]["manual_source"].lower()
    assert "rpc alias packets" in row["auth_expectations"]["manual_source"].lower()
    assert "queue" in row["auth_expectations"]["manual_source"].lower()
    assert "bytewax" in row["auth_expectations"]["manual_source"].lower()
    assert any("luci openapi --json" in cmd for cmd in row["next_commands"])
    assert any("luci payload-archive-status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api active goal --json" in cmd for cmd in row["next_commands"])
    assert any("luci api canon current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api canon versions --json" in cmd for cmd in row["next_commands"])
    assert any("luci api route catalog --json" in cmd for cmd in row["next_commands"])
    assert any("luci api root law docs --json" in cmd for cmd in row["next_commands"])
    assert any("luci api cli process receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api flow receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api test execution receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model registry current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api provider current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api workflow current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api capability current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api sheet current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model routing current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model routing blockers --json" in cmd for cmd in row["next_commands"])
    assert any("luci api prompt catalog status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw source --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw scan --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw read-queue --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw note --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw receipt --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw adapter --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw candidate --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book raw training --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bytewax compact windows --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bytewax raw windows --json" in cmd for cmd in row["next_commands"])
    assert any("luci api cloud packet --work-order-id 00000000-0000-0000-0000-000000000000 --json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc cloud-packet --work-order-id 00000000-0000-0000-0000-000000000000 --json" in cmd for cmd in row["next_commands"])
    assert any("luci api model registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api provider registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci api provider registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api capability registry --json" in cmd for cmd in row["next_commands"])
    assert any("luci api capability registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api workflow registry raw --json" in cmd for cmd in row["next_commands"])
    assert any("luci api root orchestrator current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api manual current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api daemon status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api indy queue --json" in cmd for cmd in row["next_commands"])
    assert any("luci api indy responses --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bytewax windows --json" in cmd for cmd in row["next_commands"])
    assert any("luci bytewax raw windows --json" in cmd for cmd in row["next_commands"])
    assert any("luci api flow specs --json" in cmd for cmd in row["next_commands"])
    assert any("luci flow specs --json" in cmd for cmd in row["next_commands"])
    assert any("luci api flow receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci flow receipts --json" in cmd for cmd in row["next_commands"])
    assert any("luci api chrono current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api payload archive status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api root-law-docs --json" in cmd for cmd in row["next_commands"])
    assert any("luci root-law-docs --json" in cmd for cmd in row["next_commands"])
    assert any("luci api skill policy current --json" in cmd for cmd in row["next_commands"])
    assert any("luci skill policy current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api todo current --json" in cmd for cmd in row["next_commands"])
    assert any("luci todo current --json" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work batch --json" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work item --json" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work raw batch --json" in cmd for cmd in row["next_commands"])
    assert any("luci api ontology work raw item --json" in cmd for cmd in row["next_commands"])
    assert any("luci prompt recent --json" in cmd for cmd in row["next_commands"])
    assert any("luci prompt catalog status --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book read-queue --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book training --json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc decompose-prompt --payload-json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc file-prompt --payload-json" in cmd for cmd in row["next_commands"])
    assert any("luci api rpc link-prompt --payload-json" in cmd for cmd in row["next_commands"])
    assert any("luci sheet current --json" in cmd for cmd in row["next_commands"])
    assert any("scripts/luci_todo.py --json" in cmd for cmd in row["next_commands"])
    assert any("scripts/ontology_work_compiler.py --json --text" in cmd for cmd in row["next_commands"])
    assert any("scripts/test_receipt_gate.py run --scope policy_and_retirement" in cmd for cmd in row["next_commands"])
    assert any("canon_versions?limit=5" in cmd for cmd in row["next_commands"])
    assert any("model_registry?limit=20" in cmd for cmd in row["next_commands"])
    assert any("luci api bible edges --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible manuals --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible nodes --manual-id RUNTIME_GOVERNOR --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible route catalog --json" in cmd for cmd in row["next_commands"])
    assert any("luci api bible subtree --root-id 1.0.0 --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book adapter --json" in cmd for cmd in row["next_commands"])
    assert any("luci api book candidate --json" in cmd for cmd in row["next_commands"])
