#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_help_mentions_unified_rails():
    proc = subprocess.run(
        [str(ROOT / "luci"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "luci operator-route --raw-command" in proc.stdout
    assert "luci provider groq-chat --prompt" in proc.stdout
    assert "luci model admission [--run-diogenes-gate]" in proc.stdout
    assert "luci learn --text TEXT [--run-id ID] [--artifact PATH] [--candidate-kind KIND] [--json]" in proc.stdout
    assert "luci help|/help|commands [--json] [--base-url URL]" in proc.stdout
    assert "luci openapi [--json] [--base-url URL]" in proc.stdout
    assert proc.stdout.count("luci openapi [--json] [--base-url URL]") == 1
    assert "luci manual|/manual [--json] [--base-url URL]" in proc.stdout
    assert "luci manual current [--json] [--base-url URL]" in proc.stdout
    assert "luci active goal [--json] [--base-url URL]" in proc.stdout
    assert "luci root orchestrator current [--json] [--base-url URL]" in proc.stdout
    assert "luci todo [--json] [--base-url URL]" in proc.stdout
    assert "luci todo current [--json] [--base-url URL]" in proc.stdout
    assert "luci root-orchestrator [--json] [--base-url URL]" in proc.stdout
    assert "luci canon current [--json] [--base-url URL]" in proc.stdout
    assert "luci api canon current [--json] [--base-url URL]" in proc.stdout
    assert "luci skill policy current [--json] [--base-url URL]" in proc.stdout
    assert "luci chrono current [--json] [--base-url URL]" in proc.stdout
    assert "luci model-routing-current [--json] [--base-url URL]" in proc.stdout
    assert "luci model routing current [--json] [--base-url URL]" in proc.stdout
    assert "luci model-routing-blockers [--json] [--base-url URL]" in proc.stdout
    assert "luci model routing blockers [--json] [--base-url URL]" in proc.stdout
    assert "luci model registry [--json] [--base-url URL]" in proc.stdout
    assert "luci model registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci model registry current [--json] [--base-url URL]" in proc.stdout
    assert "luci capability current [--json] [--base-url URL]" in proc.stdout
    assert "luci capability registry [--json] [--base-url URL]" in proc.stdout
    assert "luci capability registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci provider current [--json] [--base-url URL]" in proc.stdout
    assert "luci provider registry [--json] [--base-url URL]" in proc.stdout
    assert "luci provider registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci workflow registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci workflow registry [--json] [--base-url URL]" in proc.stdout
    assert "luci api workflow registry [--json] [--base-url URL]" in proc.stdout
    assert "luci api workflow registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci api root law docs [--json] [--base-url URL]" in proc.stdout
    assert "luci api root-law-docs [--json] [--base-url URL]" in proc.stdout
    assert "luci api manual current [--json] [--base-url URL]" in proc.stdout
    assert "luci api active goal [--json] [--base-url URL]" in proc.stdout
    assert "luci api root orchestrator current [--json] [--base-url URL]" in proc.stdout
    assert "luci api daemon status [--json] [--base-url URL]" in proc.stdout
    assert "luci api chrono current [--json] [--base-url URL]" in proc.stdout
    assert "luci api model routing current [--json] [--base-url URL]" in proc.stdout
    assert "luci api model routing blockers [--json] [--base-url URL]" in proc.stdout
    assert "luci api model registry current [--json] [--base-url URL]" in proc.stdout
    assert "luci api model registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci api provider current [--json] [--base-url URL]" in proc.stdout
    assert "luci api provider registry [--json] [--base-url URL]" in proc.stdout
    assert "luci api provider registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci api capability current [--json] [--base-url URL]" in proc.stdout
    assert "luci api capability registry [--json] [--base-url URL]" in proc.stdout
    assert "luci api capability registry raw [--json] [--base-url URL]" in proc.stdout
    assert "luci api workflow current [--json] [--base-url URL]" in proc.stdout
    assert "luci api indy queue [--json] [--base-url URL]" in proc.stdout
    assert "luci api indy responses [--json] [--base-url URL]" in proc.stdout
    assert "luci api todo current [--json] [--base-url URL]" in proc.stdout
    assert "luci api sheet current [--json]" in proc.stdout
    assert "luci api flow specs [--json] [--base-url URL]" in proc.stdout
    assert "luci api flow receipts [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt recent [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt filed [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompts filed [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt links [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt work-order links [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt unlinked [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt catalog [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt catalog status [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt raw recent [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt raw filed [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt raw links [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt raw unlinked [--json] [--base-url URL]" in proc.stdout
    assert "luci api prompt raw catalog [--json] [--base-url URL]" in proc.stdout
    assert "luci api book source [--json] [--base-url URL]" in proc.stdout
    assert "luci api book scan [--json] [--base-url URL]" in proc.stdout
    assert "luci api book read-queue [--json] [--base-url URL]" in proc.stdout
    assert "luci api book read queue [--json] [--base-url URL]" in proc.stdout
    assert "luci api book note [--json] [--base-url URL]" in proc.stdout
    assert "luci api book candidate [--json] [--base-url URL]" in proc.stdout
    assert "luci api book adapter [--json] [--base-url URL]" in proc.stdout
    assert "luci api book training [--json] [--base-url URL]" in proc.stdout
    assert "luci api book receipt [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw source [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw scan [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw read-queue [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw note [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw candidate [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw adapter [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw training [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw receipt [--json] [--base-url URL]" in proc.stdout
    assert "luci api ontology work batch [--json] [--base-url URL]" in proc.stdout
    assert "luci api ontology work item [--json] [--base-url URL]" in proc.stdout
    assert "luci api ontology work raw batch [--json] [--base-url URL]" in proc.stdout
    assert "luci api ontology work raw item [--json] [--base-url URL]" in proc.stdout
    assert "luci api rpc subtree --root-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci api rpc sort-key --node-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci api rpc material --node-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci api rpc cloud-packet --work-order-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci api rpc file-prompt --payload-json JSON [--json] [--base-url URL]" in proc.stdout
    assert "luci api rpc decompose-prompt --payload-json JSON [--json] [--base-url URL]" in proc.stdout
    assert "luci api rpc link-prompt --payload-json JSON [--json] [--base-url URL]" in proc.stdout
    assert "luci api cli process receipts [--json] [--base-url URL]" in proc.stdout
    assert "luci api payload archive status [--json] [--base-url URL]" in proc.stdout
    assert "luci api cloud packet --work-order-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci api bytewax windows [--json] [--base-url URL]" in proc.stdout
    assert "luci api bytewax compact windows [--json] [--base-url URL]" in proc.stdout
    assert "luci api bytewax raw windows [--json] [--base-url URL]" in proc.stdout
    assert "luci api book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in proc.stdout
    assert "luci api book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in proc.stdout
    assert "luci api ontology work <batch|item> [--json] [--base-url URL]" in proc.stdout
    assert "luci api ontology work raw <batch|item> [--json] [--base-url URL]" in proc.stdout
    assert "luci api route catalog [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible manuals [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible route catalog [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible edges [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible nodes --manual-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible subtree --root-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci rpc subtree --root-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci rpc sort-key --node-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci rpc material --node-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci rpc cloud-packet --work-order-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci rpc file-prompt --payload-json JSON [--json] [--base-url URL]" in proc.stdout
    assert "luci rpc decompose-prompt --payload-json JSON [--json] [--base-url URL]" in proc.stdout
    assert "luci rpc link-prompt --payload-json JSON [--json] [--base-url URL]" in proc.stdout
    assert "luci bible manuals [--json] [--base-url URL]" in proc.stdout
    assert "luci bible route catalog [--json] [--base-url URL]" in proc.stdout
    assert "luci bible edges [--json] [--base-url URL]" in proc.stdout
    assert "luci bible nodes --manual-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci bible subtree --root-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci daemon status [--json] [--base-url URL]" in proc.stdout
    assert "luci prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in proc.stdout
    assert "luci prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in proc.stdout
    assert "luci prompt recent [--json] [--base-url URL]" in proc.stdout
    assert "luci prompt filed [--json] [--base-url URL]" in proc.stdout
    assert "luci prompt links [--json] [--base-url URL]" in proc.stdout
    assert "luci prompt unlinked [--json] [--base-url URL]" in proc.stdout
    assert "luci prompt catalog [--json] [--base-url URL]" in proc.stdout
    assert "luci api test execution receipts [--json] [--base-url URL]" in proc.stdout
    assert "luci flow specs [--json] [--base-url URL]" in proc.stdout
    assert "luci flow receipts [--json] [--base-url URL]" in proc.stdout
    assert "luci indy queue [--json] [--base-url URL]" in proc.stdout
    assert "luci bytewax windows [--json] [--base-url URL]" in proc.stdout
    assert "luci bytewax raw windows [--json] [--base-url URL]" in proc.stdout
    assert "luci cloud packet --work-order-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in proc.stdout
    assert "luci book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in proc.stdout
    assert "luci lora candidate [--json] [--base-url URL]" in proc.stdout
    assert "luci lora adapter [--json] [--base-url URL]" in proc.stdout
    assert "luci training job [--json] [--base-url URL]" in proc.stdout
    assert "luci ontology work <batch|item> [--json] [--base-url URL]" in proc.stdout
    assert "luci ontology work raw <batch|item> [--json] [--base-url URL]" in proc.stdout
    assert "luci canon versions [--json] [--base-url URL]" in proc.stdout
    assert "luci api canon versions [--json] [--base-url URL]" in proc.stdout
    assert "luci sheet current [--json]" in proc.stdout
    assert "luci workflow current [--json] [--base-url URL]" in proc.stdout
    assert "luci cli-process-receipts [--json] [--base-url URL]" in proc.stdout
    assert "luci cli-retention [--json] [--archive-all] [--max-rows N] [--older-than-hours N]" in proc.stdout
    assert "luci payload-archive-status [--json] [--base-url URL]" in proc.stdout


def test_luci_help_json_lists_operator_commands_without_api_hard_fail():
    proc = subprocess.run(
        [str(ROOT / "luci"), "help", "--json", "--base-url", "http://127.0.0.1:9"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    commands = {row["command"] for row in payload["commands"]}
    assert payload["schema"] == "lucidota.luci.help_manual.v1"
    assert payload["mode"] == "help"
    assert payload["api_status"] == "degraded"
    assert "luci indy-response [--json]" in commands
    assert "luci /flow | luci flow ui" in commands
    assert "Phantom takeover" in payload["forbidden_detours"]


def test_luci_openapi_json_reads_live_openapi_document():
    proc = subprocess.run(
        [str(ROOT / "luci"), "openapi", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/")
    assert payload["payload"]["swagger"] == "2.0"
    assert payload["payload"]["info"]["title"] == "PostgREST API"


def test_luci_root_law_docs_json_reads_live_root_law_docs_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "root-law-docs", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "FOUND"
    assert "/root_law_docs" in payload["source_url"]
    assert payload["rows"]
    first = payload["rows"][0]
    assert first["title"] == "Root-Law API docs"
    assert first["status"] == "ok"


def test_luci_api_root_law_docs_json_reads_live_root_law_docs_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "root", "law", "docs", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "FOUND"
    assert "/api_root_law_docs" in payload["source_url"]


def test_luci_api_root_law_docs_hyphenated_alias_json_reads_live_root_law_docs_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "root-law-docs", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "FOUND"
    assert "/root_law_docs" in payload["source_url"]


def test_luci_api_prompt_raw_recent_json_reads_live_prompt_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "prompt", "raw", "recent", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "prompt_id" in row
    assert "status" in row

def test_luci_api_prompt_catalog_json_reads_live_prompt_catalog_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "prompt", "catalog", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "prompt_count" in row


def test_luci_api_prompts_filed_json_reads_live_prompt_filed_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "prompts", "filed", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_api_prompt_work_order_links_json_reads_live_prompt_link_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "prompt", "work-order", "links", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_api_prompt_catalog_status_json_reads_live_prompt_catalog_status_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "prompt", "catalog", "status", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_api_book_read_queue_json_reads_live_book_read_queue_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "book", "read", "queue", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_api_bytewax_compact_windows_json_reads_live_bytewax_compact_windows_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "bytewax", "compact", "windows", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_model_registry_raw_json_reads_live_model_registry_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "model", "registry", "raw", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict)
    assert payload["ok"] is True
    assert payload["status"] == "FOUND"
    assert payload["rows"]


def test_luci_api_model_registry_raw_json_reads_live_model_registry_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "model", "registry", "raw", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict)
    assert payload["ok"] is True
    assert payload["status"] == "FOUND"
    assert payload["rows"]


def test_luci_api_cli_process_receipts_json_reads_live_cli_receipts_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "cli", "process", "receipts", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "receipt_uuid" in row


def test_luci_api_payload_archive_status_json_reads_live_archive_status_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "payload", "archive", "status", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "source_table" in row


def test_luci_api_cloud_packet_json_reads_live_cloud_packet():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "cloud", "packet", "--work-order-id", "00000000-0000-0000-0000-000000000000", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict)


def test_luci_api_bytewax_windows_json_reads_live_bytewax_windows_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "bytewax", "windows", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "compact_window_uuid" in row


def test_luci_api_book_raw_source_json_reads_live_book_source_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "book", "raw", "source", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_api_ontology_work_raw_batch_json_reads_live_ontology_batch_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "ontology", "work", "raw", "batch", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload


def test_luci_api_bible_subtree_route_is_live_for_root_node():
    with urllib.request.urlopen("http://127.0.0.1:3000/api_bible_subtree?root_id=eq.1.0.0&limit=1", timeout=8) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))
    assert isinstance(payload, list) and payload
    first = payload[0]
    assert first["root_id"] == "1.0.0"
    assert first["node_id"] == "1.0.0"


def test_luci_help_surface_adds_root_manual_command_when_route_is_live(monkeypatch):
    import scripts.luci_help_manual as help_manual

    def fake_fetch_json(base_url: str, path: str, query: dict[str, str] | None = None):
        if path == "":
            return True, {
                "paths": {
                    "/manual_current": {"get": {}},
                    "/active_goal": {"get": {}},
                    "/root_orchestrator_current": {"get": {}},
                    "/canon_current": {"get": {}},
                    "/root_law_docs": {"get": {}},
                    "/root_orchestrator_current": {"get": {}},
                    "/chrono_current": {"get": {}},
                    "/model_routing_current": {"get": {}},
                    "/model_registry_current": {"get": {}},
                    "/skill_policy_current": {"get": {}},
                    "/capability_current": {"get": {}},
                    "/provider_current": {"get": {}},
                    "/provider_registry": {"get": {}},
                    "/api_workflow_registry": {"get": {}},
                    "/api_route_catalog": {"get": {}},
                    "/api_bible_manuals": {"get": {}},
                    "/api_bible_route_catalog": {"get": {}},
                    "/api_bible_edges": {"get": {}},
                    "/api_bible_nodes": {"get": {}},
                    "/api_bible_subtree": {"get": {}},
                    "/flow_specs": {"get": {}},
                    "/flow_receipts": {"get": {}},
                    "/capability_registry": {"get": {}},
                    "/daemon_status": {"get": {}},
                    "/prompt_recent": {"get": {}},
                    "/prompts_filed": {"get": {}},
                    "/prompt_work_order_links": {"get": {}},
                    "/prompt_unlinked": {"get": {}},
                    "/prompt_catalog_status": {"get": {}},
                    "/indy_queue": {"get": {}},
                    "/indy_responses": {"get": {}},
                    "/bytewax_compact_windows": {"get": {}},
                    "/rpc/cloud_packet": {"post": {}},
                    "/rpc/get_subtree": {"get": {}},
                    "/rpc/fn_bible_node_sort_key": {"get": {}},
                    "/rpc/fn_bible_node_material": {"post": {}},
                    "/book_source": {"get": {}},
                    "/book_scan": {"get": {}},
                    "/book_read_queue": {"get": {}},
                    "/book_note": {"get": {}},
                    "/lora_candidate": {"get": {}},
                    "/lora_adapter": {"get": {}},
                    "/training_job": {"get": {}},
                    "/book_receipt": {"get": {}},
                    "/ontology_work_batch": {"get": {}},
                    "/ontology_work_item": {"get": {}},
                    "/canon_versions": {"get": {}},
                    "/sheet_current": {"get": {}},
                    "/workflow_current": {"get": {}},
                }
            }, f"{base_url}/"
        if path == "manual_current":
            return True, [{"manual_id": "LUCIDOTA_OPERATOR_MANUAL", "route_list": [], "live_surface": {}}], f"{base_url}/manual_current"
        return True, [], f"{base_url}/{path}"

    monkeypatch.setattr(help_manual, "fetch_json", fake_fetch_json)
    payload = help_manual.build_payload("help", "http://127.0.0.1:3000")
    commands = {row["command"] for row in payload["commands"]}
    assert "luci root-orchestrator [--json] [--base-url URL]" in commands
    assert "luci manual current [--json] [--base-url URL]" in commands
    assert "luci active goal [--json] [--base-url URL]" in commands
    assert "luci root orchestrator current [--json] [--base-url URL]" in commands
    assert "luci canon current [--json] [--base-url URL]" in commands
    assert "luci api canon current [--json] [--base-url URL]" in commands
    assert "luci skill policy current [--json] [--base-url URL]" in commands
    assert "luci chrono current [--json] [--base-url URL]" in commands
    assert "luci model-routing-current [--json] [--base-url URL]" in commands
    assert "luci model routing current [--json] [--base-url URL]" in commands
    assert "luci model-routing-blockers [--json] [--base-url URL]" in commands
    assert "luci model routing blockers [--json] [--base-url URL]" in commands
    assert "luci model registry [--json] [--base-url URL]" in commands
    assert "luci model registry current [--json] [--base-url URL]" in commands
    assert "luci capability current [--json] [--base-url URL]" in commands
    assert "luci capability registry [--json] [--base-url URL]" in commands
    assert "luci capability registry raw [--json] [--base-url URL]" in commands
    assert "luci provider current [--json] [--base-url URL]" in commands
    assert "luci provider registry [--json] [--base-url URL]" in commands
    assert "luci provider registry raw [--json] [--base-url URL]" in commands
    assert "luci workflow registry raw [--json] [--base-url URL]" in commands
    assert "luci workflow registry [--json] [--base-url URL]" in commands
    assert "luci api workflow registry [--json] [--base-url URL]" in commands
    assert "luci api workflow registry raw [--json] [--base-url URL]" in commands
    assert "luci api root law docs [--json] [--base-url URL]" in commands
    assert "luci api prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in commands
    assert "luci api prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in commands
    assert "luci api prompt catalog [--json] [--base-url URL]" in commands
    assert "luci api cloud packet --work-order-id ID [--json] [--base-url URL]" in commands
    assert "luci api bytewax windows [--json] [--base-url URL]" in commands
    assert "luci api bytewax raw windows [--json] [--base-url URL]" in commands
    assert "luci api book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in commands
    assert "luci api book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in commands
    assert "luci api ontology work <batch|item> [--json] [--base-url URL]" in commands
    assert "luci api ontology work raw <batch|item> [--json] [--base-url URL]" in commands
    assert "luci api route catalog [--json] [--base-url URL]" in commands
    assert "luci api bible manuals [--json] [--base-url URL]" in commands
    assert "luci api bible route catalog [--json] [--base-url URL]" in commands
    assert "luci api bible edges [--json] [--base-url URL]" in commands
    assert "luci api bible nodes --manual-id ID [--json] [--base-url URL]" in commands
    assert "luci api bible subtree --root-id ID [--json] [--base-url URL]" in commands
    assert "luci rpc subtree --root-id ID [--json] [--base-url URL]" in commands
    assert "luci rpc sort-key --node-id ID [--json] [--base-url URL]" in commands
    assert "luci rpc material --node-id ID [--json] [--base-url URL]" in commands
    assert "luci rpc cloud-packet --work-order-id ID [--json] [--base-url URL]" in commands
    assert "luci rpc file-prompt --payload-json JSON [--json] [--base-url URL]" in commands
    assert "luci rpc decompose-prompt --payload-json JSON [--json] [--base-url URL]" in commands
    assert "luci rpc link-prompt --payload-json JSON [--json] [--base-url URL]" in commands
    assert "luci bible manuals [--json] [--base-url URL]" in commands
    assert "luci bible route catalog [--json] [--base-url URL]" in commands
    assert "luci bible edges [--json] [--base-url URL]" in commands
    assert "luci bible nodes --manual-id ID [--json] [--base-url URL]" in commands
    assert "luci bible subtree --root-id ID [--json] [--base-url URL]" in commands
    assert "luci daemon status [--json] [--base-url URL]" in commands
    assert "luci prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in commands
    assert "luci prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]" in commands
    assert "luci prompt recent [--json] [--base-url URL]" in commands
    assert "luci prompt filed [--json] [--base-url URL]" in commands
    assert "luci prompt links [--json] [--base-url URL]" in commands
    assert "luci prompt unlinked [--json] [--base-url URL]" in commands
    assert "luci prompt catalog [--json] [--base-url URL]" in commands
    assert "luci api test execution receipts [--json] [--base-url URL]" in commands
    assert "luci api route catalog [--json] [--base-url URL]" in commands
    assert "luci flow specs [--json] [--base-url URL]" in commands
    assert "luci flow receipts [--json] [--base-url URL]" in commands
    assert "luci indy queue [--json] [--base-url URL]" in commands
    assert "luci bytewax windows [--json] [--base-url URL]" in commands
    assert "luci bytewax raw windows [--json] [--base-url URL]" in commands
    assert "luci cloud packet --work-order-id ID [--json] [--base-url URL]" in commands
    assert "luci book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in commands
    assert "luci book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]" in commands
    assert "luci lora candidate [--json] [--base-url URL]" in commands
    assert "luci lora adapter [--json] [--base-url URL]" in commands
    assert "luci training job [--json] [--base-url URL]" in commands
    assert "luci ontology work <batch|item> [--json] [--base-url URL]" in commands
    assert "luci ontology work raw <batch|item> [--json] [--base-url URL]" in commands
    assert "luci canon versions [--json] [--base-url URL]" in commands
    assert "luci api canon versions [--json] [--base-url URL]" in commands
    assert "luci todo current [--json] [--base-url URL]" in commands
    assert "luci sheet current --json" in commands
    assert "luci workflow current --json" in commands or "luci workflow current [--json] [--base-url URL]" in commands
    assert "luci root-law-docs [--json] [--base-url URL]" in commands


def test_luci_root_orchestrator_json_shortcut_stays_thin():
    proc = subprocess.run(
        [str(ROOT / "luci"), "root-orchestrator", "--json", "--base-url", "http://127.0.0.1:9"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ROUTE_NOT_LIVE"
    assert payload["source_url"].endswith("/root_orchestrator_current")


def test_luci_root_orchestrator_current_json_reads_live_root_orchestrator_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "root", "orchestrator", "current", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/root_orchestrator_current?limit=1")
    assert isinstance(payload["rows"], list)
    row = payload["rows"][0]
    assert row["orchestrator_id"] == "ROOT_ORCHESTRATOR_CURRENT"
    assert "route_count" in row
    assert "route_list" in row


def test_luci_api_test_execution_receipts_json_reads_live_receipts_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "test", "execution", "receipts", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "receipt_uuid" in row
    assert "scope" in row
    assert "status" in row


def test_luci_api_route_catalog_json_reads_live_route_catalog_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "route", "catalog", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "route_id" in row
    assert "path_pattern" in row
    assert "status" in row


def test_luci_flow_specs_json_reads_live_flow_specs_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "flow", "specs", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "flow_id" in row
    assert "name" in row
    assert "status" in row


def test_luci_flow_receipts_json_reads_live_flow_receipts_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "flow", "receipts", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "receipt_id" in row
    assert "flow_id" in row
    assert "status" in row


def test_manual_current_mentions_api_test_execution_receipts():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "api_test_execution_receipts" in route_ids


def test_manual_current_mentions_api_route_catalog():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "api_route_catalog" in route_ids


def test_manual_current_mentions_api_bible_nodes():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "nodes" in route_ids




def test_manual_current_mentions_api_bible_manuals():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "manuals" in route_ids


def test_luci_bible_manuals_json_reads_live_bible_manuals_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "bible", "manuals", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["source_url"].endswith("/api_bible_manuals?order=manual_id.asc&limit=20")
    assert isinstance(payload["payload"], list)
    assert payload["payload"]
    row = payload["payload"][0]
    assert "manual_id" in row
    assert "node_count" in row


def test_manual_current_mentions_api_bible_route_catalog():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "route_catalog" in route_ids


def test_luci_bible_route_catalog_json_reads_live_bible_route_catalog_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "bible", "route", "catalog", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["source_url"].endswith("/api_bible_route_catalog?order=route_id.asc&limit=20")
    assert isinstance(payload["payload"], list)
    assert payload["payload"]
    row = payload["payload"][0]
    assert "route_id" in row
    assert "path_pattern" in row


def test_manual_current_mentions_api_bible_edges():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "edges" in route_ids


def test_luci_bible_edges_json_reads_live_bible_edges_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "bible", "edges", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["source_url"].endswith("/api_bible_edges?order=edge_id.asc&limit=20")
    assert isinstance(payload["payload"], list)
    # may be empty if no edges, but route should be reachable
def test_manual_current_mentions_flow_specs_and_receipts():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "flow_specs" in route_ids
    assert "flow_receipts" in route_ids


def test_manual_current_mentions_rpc_bible_helpers():
    with urllib.request.urlopen("http://127.0.0.1:3000/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    row = payload[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert "get_subtree" in route_ids
    assert "fn_bible_node_sort_key" in route_ids
    assert "fn_bible_node_material" in route_ids


def test_luci_bible_nodes_json_reads_live_bible_nodes_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "bible", "nodes", "--manual-id", "RUNTIME_GOVERNOR", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["source_url"].endswith("/api_bible_nodes?manual_id=eq.RUNTIME_GOVERNOR&order=node_sort_key.asc&limit=20")
    assert isinstance(payload["payload"], list)
    assert payload["payload"]
    row = payload["payload"][0]
    assert row["manual_id"] == "RUNTIME_GOVERNOR"
    assert "node_id" in row


def test_luci_canon_current_json_reads_live_canon_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "canon", "current", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "node_id" in row
    assert "title" in row
    assert "status" in row
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert any("luci canon current --json" in cmd for cmd in row["next_commands"])


def test_luci_rpc_bible_helpers_json_reads_live_rpc_surfaces():
    subtree = subprocess.run(
        [str(ROOT / "luci"), "rpc", "subtree", "--root-id", "4.9511.0", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    subtree_payload = json.loads(subtree.stdout)
    assert subtree_payload["source_url"].endswith("/rpc/get_subtree?root_id=4.9511.0")
    assert isinstance(subtree_payload["payload"], list)
    assert subtree_payload["payload"]
    assert subtree_payload["payload"][0]["node_id"] == "4.9511.0"

    sort_key = subprocess.run(
        [str(ROOT / "luci"), "rpc", "sort-key", "--node-id", "4.9511.0", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    sort_key_payload = json.loads(sort_key.stdout)
    assert sort_key_payload["source_url"].endswith("/rpc/fn_bible_node_sort_key?p_node_id=4.9511.0")
    assert sort_key_payload["payload"] == [4, 9511, 0]

    material = subprocess.run(
        [str(ROOT / "luci"), "rpc", "material", "--node-id", "1.0.0", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    material_payload = json.loads(material.stdout)
    assert material_payload["source_url"].endswith("/rpc/fn_bible_node_material")
    assert material_payload["payload"]["node_id"] == "1.0.0"
    assert "title" in material_payload["payload"]


def test_luci_active_goal_json_reads_live_active_goal_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "active", "goal", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "FOUND"
    assert payload["source_url"].endswith("/active_goal?limit=1")
    assert isinstance(payload["rows"], list)
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["goal_id"] == "indy-response-out-loop"
    assert row["status"] == "active"
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert row["db_law"]["statement"].startswith("Postgres/PostgREST is truth")
    assert isinstance(row.get("next_commands"), list)
    assert any("luci active goal --json" in cmd for cmd in row["next_commands"])


def test_luci_chrono_current_json_reads_live_chrono_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "chrono", "current", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "chrono_packet_id" in row
    assert "prompt_ledger" in row
    assert "work_ledger" in row


def test_luci_manual_json_uses_manual_current_surface_without_api_hard_fail():
    proc = subprocess.run(
        [str(ROOT / "luci"), "/manual", "--json", "--base-url", "http://127.0.0.1:9"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "manual"
    assert payload["source"] == "postgrest_safe_surface"
    assert payload["manual_current_url"].startswith("http://127.0.0.1:9/manual_current")


def test_luci_manual_current_json_reads_live_manual_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "manual", "current", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "manual"
    assert payload["source"] == "postgrest_safe_surface"
    assert "/manual_current" in payload["manual_current_url"]


def test_luci_workflow_registry_json_reads_live_workflow_registry_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "workflow", "registry", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "workflow_id" in row
    assert "workflow_name" in row
    assert "status" in row


def test_luci_api_workflow_registry_json_reads_live_workflow_registry_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "api", "workflow", "registry", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "workflow_id" in row
    assert "workflow_name" in row


def test_luci_daemon_status_json_reads_live_daemon_status_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "daemon", "status", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "daemon_name" in row
    assert "heartbeat_kind" in row
    assert "process_id" in row
    assert "goal" in row
    assert "db_law" in row
    assert "next_commands" in row


def test_luci_provider_registry_json_reads_live_provider_registry_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "provider", "registry", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "provider_key" in row
    assert "provider_kind" in row


def test_luci_capability_registry_json_reads_live_capability_registry_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "capability", "registry", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "capability_key" in row
    assert "capability_group" in row


def test_luci_prompt_recent_json_reads_live_prompt_recent_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "prompt", "recent", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "prompt_id" in row
    assert "raw_prompt_text" in row


def test_luci_indy_queue_json_reads_live_indy_queue_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "indy", "queue", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "id" in row or "event_id" in row
    assert "processed_status" in row


def test_luci_bytewax_windows_json_reads_live_bytewax_windows_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "bytewax", "windows", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "compact_window_uuid" in row
    assert "window_kind" in row


def test_luci_bytewax_raw_windows_json_reads_live_bytewax_windows_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "bytewax", "raw", "windows", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "compact_window_uuid" in row
    assert "window_kind" in row


def test_luci_book_source_json_reads_live_book_source_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "book", "source", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_book_raw_source_json_reads_live_book_source_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "book", "raw", "source", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    if payload:
        row = payload[0]
        assert "book_uuid" in row or "source_uuid" in row or "graph_item_uuid" in row


def test_luci_prompt_raw_recent_json_reads_live_prompt_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "prompt", "raw", "recent", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "prompt_id" in row
    assert "status" in row


def test_luci_ontology_work_batch_json_reads_live_ontology_batch_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "ontology", "work", "batch", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "batch_uuid" in row or "batch_key" in row


def test_luci_ontology_work_raw_batch_json_reads_live_ontology_batch_rows():
    proc = subprocess.run(
        [str(ROOT / "luci"), "ontology", "work", "raw", "batch", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "batch_uuid" in row or "batch_key" in row


def test_luci_lora_candidate_json_reads_live_candidate_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "lora", "candidate", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_lora_adapter_json_reads_live_adapter_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "lora", "adapter", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_training_job_json_reads_live_training_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "training", "job", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)


def test_luci_canon_versions_json_reads_live_canon_versions_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "canon", "versions", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "version_id" in row or "node_id" in row or "canon_id" in row


def test_luci_cli_process_receipts_json_reads_live_receipt_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "cli-process-receipts", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert "receipt_uuid" in row
    assert "status" in row
    assert "auth_injected" in row


def test_luci_payload_archive_status_json_reads_live_archive_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "payload-archive-status", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload == [] or "source_table" in payload[0]


def test_luci_todo_json_reads_live_queue_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "todo", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["batch_key"].startswith("ontobatch:")
    assert "goal" in payload[0]
    assert "db_law" in payload[0]
    assert "next_commands" in payload[0]


def test_luci_todo_current_json_reads_live_queue_surface():
    proc = subprocess.run(
        [str(ROOT / "luci"), "todo", "current", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["batch_key"].startswith("ontobatch:")
    assert "goal" in payload[0]
    assert "db_law" in payload[0]
    assert "next_commands" in payload[0]


def test_luci_stdin_help_shortcut_stays_out_of_model_engine():
    proc = subprocess.run(
        [str(ROOT / "luci")],
        cwd=ROOT,
        input="/help\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert "LUCI HELP" in proc.stdout
    assert "Commands:" in proc.stdout
    assert "REPORT_PATH=05_OUTPUTS/luci/" not in proc.stdout
