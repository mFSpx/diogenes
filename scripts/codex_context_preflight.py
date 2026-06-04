#!/usr/bin/env python3
"""LUCIDOTA Codex preflight: print live PostgREST/manual truth first."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("POSTGREST_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
ROUTES = [
    "manual_current",
    "canon_current",
    "canon_versions",
    "active_goal",
    "api_route_catalog",
    "root_law_docs",
    "skill_policy_current",
    "root_orchestrator_current",
    "chrono_current",
    "ontology_work_batch",
    "ontology_work_item",
    "todo_current",
    "bytewax_compact_windows",
    "api_workflow_registry",
    "api_test_execution_receipts",
    "api_bible_manuals",
    "api_bible_route_catalog",
    "api_bible_edges",
    "api_bible_nodes",
    "api_bible_subtree",
    "rpc/get_subtree",
    "rpc/fn_bible_node_sort_key",
    "rpc/fn_bible_node_material",
    "flow_specs",
    "flow_receipts",
    "capability_registry",
    "capability_current",
    "provider_current",
    "model_registry",
    "model_registry_current",
    "model_routing_current",
    "model_routing_blockers",
    "provider_registry",
    "workflow_registry",
    "workflow_current",
    "daemon_status",
    "indy_queue",
    "indy_responses",
    "flow_specs",
    "flow_receipts",
    "prompts_filed",
    "prompt_work_order_links",
    "prompt_recent",
    "prompt_unlinked",
    "prompt_catalog_status",
    "chrono_current",
    "sheet_current",
    "book_source",
    "book_scan",
    "book_read_queue",
    "book_note",
    "lora_candidate",
    "lora_adapter",
    "training_job",
    "book_receipt",
]


def fetch_json(path: str, query: dict[str, str] | None = None) -> tuple[int, Any, str]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{BASE_URL}/{path}" + (f"?{qs}" if qs else "")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null"), ""
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        return exc.code, None, body
    except Exception as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def post_json(path: str, payload: dict[str, Any]) -> tuple[int, Any, str]:
    url = f"{BASE_URL}/{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null"), ""
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        return exc.code, None, body
    except Exception as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def route_status(path: str, openapi_paths: dict[str, Any]) -> dict[str, Any]:
    status, body, error = fetch_json(path, {"limit": "3"})
    rows = body if isinstance(body, list) else []
    return {
        "route": f"/{path}",
        "openapi_methods": sorted(openapi_paths.get(f"/{path}", {}).keys()),
        "http_status": status,
        "sample_rows": len(rows),
        "fields": sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else [],
        "error": error,
    }


def main() -> int:
    openapi_status, openapi, openapi_error = fetch_json("", None)
    openapi_paths = openapi.get("paths", {}) if isinstance(openapi, dict) else {}
    routes = [route_status(route, openapi_paths) for route in ROUTES]
    cloud_packet_status, cloud_packet_body, cloud_packet_error = post_json(
        "rpc/cloud_packet",
        {
            "work_order_id": "00000000-0000-0000-0000-000000000000",
            "max_chars": 256,
            "max_items": 1,
            "task_type": "preflight",
            "target_model": "preflight",
            "include_raw_bodies": False,
        },
    )
    cloud_packet_route = {
        "route": "/rpc/cloud_packet",
        "openapi_methods": sorted(openapi_paths.get("/rpc/cloud_packet", {}).keys()),
        "http_status": cloud_packet_status,
        "sample_rows": 1 if isinstance(cloud_packet_body, dict) else 0,
        "fields": sorted(cloud_packet_body.keys()) if isinstance(cloud_packet_body, dict) else [],
        "error": cloud_packet_error,
    }
    active_rows = next((r for r in routes if r["route"] == "/active_goal"), {})
    active_status, active_body, _ = fetch_json("active_goal", {"limit": "1"})
    active_goal = active_body[0] if active_status == 200 and isinstance(active_body, list) and active_body else None
    indy_queue = next((r for r in routes if r["route"] == "/indy_queue"), {})
    indy_responses = next((r for r in routes if r["route"] == "/indy_responses"), {})
    root_law_docs = next((r for r in routes if r["route"] == "/root_law_docs"), {})
    api_bible_subtree = next((r for r in routes if r["route"] == "/api_bible_subtree"), {})

    report = {
        "schema": "lucidota.codex_context_preflight.v1",
        "product_root": str(ROOT),
        "postgrest_base_url": BASE_URL,
        "openapi_status": openapi_status,
        "openapi_path_count": len(openapi_paths),
        "openapi_error": openapi_error,
        "current_task": active_goal
        or {
            "title": "Indy response-out loop",
            "source": "operator_supplied_current_task; /active_goal currently empty",
            "active_goal_route_status": active_rows.get("http_status"),
        },
        "verified_working_state": {
            "postgrest_openapi": openapi_status == 200,
            "manual_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/manual_current"), False),
            "root_orchestrator_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/root_orchestrator_current"), False),
            "prompt_catalog_status": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompt_catalog_status"), False),
            "prompt_recent": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompt_recent"), False),
            "prompt_unlinked": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompt_unlinked"), False),
            "prompts_filed": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompts_filed"), False),
            "chrono_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/chrono_current"), False),
            "sheet_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/sheet_current"), False),
            "skill_policy_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/skill_policy_current"), False),
            "chrono_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/chrono_current"), False),
            "model_routing_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/model_routing_current"), False),
            "model_registry_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/model_registry_current"), False),
            "capability_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/capability_current"), False),
            "provider_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/provider_current"), False),
            "api_workflow_registry": next((r["http_status"] == 200 for r in routes if r["route"] == "/api_workflow_registry"), False),
            "daemon_status": next((r["http_status"] == 200 for r in routes if r["route"] == "/daemon_status"), False),
            "capability_registry": next((r["http_status"] == 200 for r in routes if r["route"] == "/capability_registry"), False),
            "provider_registry": next((r["http_status"] == 200 for r in routes if r["route"] == "/provider_registry"), False),
            "prompt_recent": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompt_recent"), False),
            "prompts_filed": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompts_filed"), False),
            "prompt_work_order_links": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompt_work_order_links"), False),
            "prompt_unlinked": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompt_unlinked"), False),
            "prompt_catalog_status": next((r["http_status"] == 200 for r in routes if r["route"] == "/prompt_catalog_status"), False),
            "indy_queue": next((r["http_status"] == 200 for r in routes if r["route"] == "/indy_queue"), False),
            "indy_responses": next((r["http_status"] == 200 for r in routes if r["route"] == "/indy_responses"), False),
            "bytewax_compact_windows": next((r["http_status"] == 200 for r in routes if r["route"] == "/bytewax_compact_windows"), False),
            "book_source": next((r["http_status"] == 200 for r in routes if r["route"] == "/book_source"), False),
            "book_scan": next((r["http_status"] == 200 for r in routes if r["route"] == "/book_scan"), False),
            "book_read_queue": next((r["http_status"] == 200 for r in routes if r["route"] == "/book_read_queue"), False),
            "book_note": next((r["http_status"] == 200 for r in routes if r["route"] == "/book_note"), False),
            "lora_candidate": next((r["http_status"] == 200 for r in routes if r["route"] == "/lora_candidate"), False),
            "lora_adapter": next((r["http_status"] == 200 for r in routes if r["route"] == "/lora_adapter"), False),
            "training_job": next((r["http_status"] == 200 for r in routes if r["route"] == "/training_job"), False),
            "book_receipt": next((r["http_status"] == 200 for r in routes if r["route"] == "/book_receipt"), False),
            "ontology_work_batch": next((r["http_status"] == 200 for r in routes if r["route"] == "/ontology_work_batch"), False),
            "ontology_work_item": next((r["http_status"] == 200 for r in routes if r["route"] == "/ontology_work_item"), False),
            "canon_versions": next((r["http_status"] == 200 for r in routes if r["route"] == "/canon_versions"), False),
            "workflow_current": next((r["http_status"] == 200 for r in routes if r["route"] == "/workflow_current"), False),
            "indy_queue_visible": indy_queue.get("http_status") == 200,
            "indy_responses_visible": indy_responses.get("http_status") == 200,
            "root_law_docs_visible": root_law_docs.get("http_status") == 200,
            "api_bible_subtree_visible": api_bible_subtree.get("http_status") == 200,
            "cloud_packet_visible": cloud_packet_status == 200,
        },
        "allowed_focus_files": [
            "scripts/indy_reads.py",
            "scripts/indy_conduit_driver.py",
            "scripts/luci_operator.py",
            "luci",
            "scripts/luci_help_manual.py",
            "scripts/prompt_ledger_capture.py",
            "06_SCHEMA/20260604_luci_product_safe_surface.sql",
            "services/ironclaw-indy-reads.service",
            "scripts/lucidota_start_indy_reads_watcher.sh",
            "04_RUNTIME/indy_reads_startup_comms_manifest.json",
            "tests/test_indy_reads_chat.py",
            "tests/test_luci_operator_indy_conduit.py",
            "tests/test_indy_startup_comms_speed.py",
            "tests/test_systemd_control_surfaces.py",
        ],
        "forbidden_detours": [
            "do_not_treat_GOALS_or_05_OUTPUTS_as_source_of_truth_without_API_pointer",
            "do_not_reopen_root_rotor_batch_theater",
            "do_not_call_this_DBOS",
            "do_not_scan_whole_repo_before_API_truth",
            "do_not_use_loose_JSON_as_runtime_queue_authority",
        ],
        "route_findings": routes,
        "cloud_packet_route": cloud_packet_route,
        "next_executable_batch": [
            "apply/reload safe API schema if /indy_responses is missing",
            "compile one messy operator batch into /todo_current",
            "file one steering prompt into /prompts_filed and link it when possible",
            "inspect /model_routing_current for missing local roles and route blockers",
            "inspect /model_routing_blockers for the explicit blocker packet and missing-role count",
            "inspect /sheet_current for spreadsheet-style operator work state",
            "inspect /workflow_current for registry state, active workflow names, and ownership breakdown",
            "inspect /capability_current for capability lanes, group breakdown, and workflow mapping",
            "inspect /model_registry_current for model counts, role breakdown, and loadout coverage",
            "inspect /provider_current for provider lanes, local/cloud choices, and active provider coverage",
            "inspect /skill_policy_current for live operator policy text and alignment rules",
            "inspect /chrono_current for prompt/work/execution history alignment",
            "inspect /api_bible_subtree for direct subtree rows keyed by root_id",
            "inspect /lora_candidate, /lora_adapter, and /training_job for raw LoRA/training rows",
            "queue one /indy row through luci execute",
            "run Indy --respond-once",
            "verify /indy_responses exposes response_id/body/status",
            "verify ./luci indy-response surfaces the PostgREST row",
            "audit /root_law_docs once the root manual surface is mounted",
            "verify /rpc/cloud_packet returns bounded prompt payloads",
            "verify /chrono_current aligns prompt/work/execution history",
            "verify prompt ledger routes and manual surface mention prompt filing law",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if openapi_status == 200 else 2


if __name__ == "__main__":
    sys.exit(main())
