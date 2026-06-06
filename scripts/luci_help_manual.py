#!/usr/bin/env python3
"""Thin LUCI help/manual surface backed by PostgREST safe routes when live."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:3000"
BASE_COMMANDS = [
    {"command": "luci help | luci /help | luci commands", "purpose": "show the operator command map"},
    {"command": "luci openapi [--json] [--base-url URL]", "purpose": "show the live PostgREST OpenAPI document at /"},
    {"command": "luci manual | luci /manual", "purpose": "show live manual volumes from /manual_current"},
    {"command": "luci manual current [--json] [--base-url URL]", "purpose": "show the live manual packet from /manual_current"},
    {"command": "luci manual capsule [--json] [--base-url URL]", "purpose": "show the on-demand manual capsule from /root_law_docs"},
    {"command": "luci doctor [--json] [--base-url URL]", "purpose": "show the live operator doctor surface"},
    {"command": "luci status [--json] [--base-url URL]", "purpose": "show the live operator status surface"},
    {"command": "luci active goal [--json] [--base-url URL]", "purpose": "show the live active goal packet from /active_goal"},
    {"command": "luci root orchestrator current [--json] [--base-url URL]", "purpose": "show the live root orchestrator packet from /root_orchestrator_current"},
    {"command": "luci todo", "purpose": "show the live ontology todo batch from /todo_current"},
    {"command": "luci todo current [--json] [--base-url URL]", "purpose": "show the live ontology todo batch from /todo_current"},
    {"command": "luci root-orchestrator [--json] [--base-url URL]", "purpose": "show the live root orchestrator packet from /root_orchestrator_current"},
    {"command": "luci canon current [--json] [--base-url URL]", "purpose": "show the live canon packet from /canon_current"},
    {"command": "luci api canon current [--json] [--base-url URL]", "purpose": "show the live canon packet from /canon_current"},
    {"command": "luci skill policy current [--json] [--base-url URL]", "purpose": "show the live skill policy packet from /skill_policy_current"},
    {"command": "luci chrono current [--json] [--base-url URL]", "purpose": "show the live chrono packet from /chrono_current"},
    {"command": "luci model-routing-current [--json] [--base-url URL]", "purpose": "show the live model routing packet from /model_routing_current"},
    {"command": "luci model routing current [--json] [--base-url URL]", "purpose": "show the live model routing packet from /model_routing_current"},
    {"command": "luci model-routing-blockers [--json] [--base-url URL]", "purpose": "show only the missing-role blockers from /model_routing_current"},
    {"command": "luci model routing blockers [--json] [--base-url URL]", "purpose": "show only the missing-role blockers from /model_routing_current"},
    {"command": "luci model registry [--json] [--base-url URL]", "purpose": "show the raw model registry rows from /model_registry"},
    {"command": "luci model registry raw [--json] [--base-url URL]", "purpose": "show the raw model registry rows from /model_registry"},
    {"command": "luci model registry current [--json] [--base-url URL]", "purpose": "show the live model registry packet from /model_registry_current"},
    {"command": "luci capability current [--json] [--base-url URL]", "purpose": "show the live capability packet from /capability_current"},
    {"command": "luci capability list [--json] [--base-url URL]", "purpose": "show the live capability list surface"},
    {"command": "luci capability registry [--json] [--base-url URL]", "purpose": "show the live capability registry packet from /capability_registry"},
    {"command": "luci capability registry raw [--json] [--base-url URL]", "purpose": "show the raw capability registry rows from /capability_registry"},
    {"command": "luci provider current [--json] [--base-url URL]", "purpose": "show the live provider packet from /provider_current"},
    {"command": "luci provider registry [--json] [--base-url URL]", "purpose": "show the live provider registry packet from /provider_registry"},
    {"command": "luci provider registry raw [--json] [--base-url URL]", "purpose": "show the raw provider registry rows from /provider_registry"},
    {"command": "luci workflow registry raw [--json] [--base-url URL]", "purpose": "show the raw workflow registry rows from /workflow_registry"},
    {"command": "luci workflow registry [--json] [--base-url URL]", "purpose": "show the live API workflow registry packet from /api_workflow_registry"},
    {"command": "luci api workflow registry [--json] [--base-url URL]", "purpose": "show the live API workflow registry packet from /api_workflow_registry"},
    {"command": "luci api workflow registry raw [--json] [--base-url URL]", "purpose": "show the raw API workflow registry rows from /api_workflow_registry"},
    {"command": "luci api root law docs [--json] [--base-url URL]", "purpose": "show the live root-law docs packet from /api_root_law_docs"},
    {"command": "luci api root-law-docs [--json] [--base-url URL]", "purpose": "show the live root-law docs packet from /api_root_law_docs"},
    {"command": "luci api manual current [--json] [--base-url URL]", "purpose": "show the live manual packet from /manual_current"},
    {"command": "luci api active goal [--json] [--base-url URL]", "purpose": "show the live active goal packet from /active_goal"},
    {"command": "luci api root orchestrator current [--json] [--base-url URL]", "purpose": "show the live root orchestrator packet from /root_orchestrator_current"},
    {"command": "luci api daemon status [--json] [--base-url URL]", "purpose": "show the live daemon status packet from /daemon_status"},
    {"command": "luci api chrono current [--json] [--base-url URL]", "purpose": "show the live chrono packet from /chrono_current"},
    {"command": "luci api model routing current [--json] [--base-url URL]", "purpose": "show the live model routing packet from /model_routing_current"},
    {"command": "luci api model routing blockers [--json] [--base-url URL]", "purpose": "show the live model routing blockers packet from /model_routing_blockers"},
    {"command": "luci api model registry current [--json] [--base-url URL]", "purpose": "show the live model registry packet from /model_registry_current"},
    {"command": "luci api model registry raw [--json] [--base-url URL]", "purpose": "show the raw model registry rows from /model_registry"},
    {"command": "luci api provider current [--json] [--base-url URL]", "purpose": "show the live provider packet from /provider_current"},
    {"command": "luci api provider registry [--json] [--base-url URL]", "purpose": "show the raw provider registry rows from /provider_registry"},
    {"command": "luci api provider registry raw [--json] [--base-url URL]", "purpose": "show the raw provider registry rows from /provider_registry"},
    {"command": "luci api capability current [--json] [--base-url URL]", "purpose": "show the live capability packet from /capability_current"},
    {"command": "luci api capability registry [--json] [--base-url URL]", "purpose": "show the raw capability registry rows from /capability_registry"},
    {"command": "luci api capability registry raw [--json] [--base-url URL]", "purpose": "show the raw capability registry rows from /capability_registry"},
    {"command": "luci api workflow current [--json] [--base-url URL]", "purpose": "show the live workflow packet from /workflow_current"},
    {"command": "luci api indy queue [--json] [--base-url URL]", "purpose": "show the live Indy queue packet from /indy_queue"},
    {"command": "luci api indy responses [--json] [--base-url URL]", "purpose": "show the live Indy response packet from /indy_responses"},
    {"command": "luci api todo current [--json] [--base-url URL]", "purpose": "show the live todo packet from /todo_current"},
    {"command": "luci api sheet current [--json]", "purpose": "show the live spreadsheet packet from /sheet_current"},
    {"command": "luci api flow specs [--json] [--base-url URL]", "purpose": "show the live flow specs packet from /flow_specs"},
    {"command": "luci api flow receipts [--json] [--base-url URL]", "purpose": "show the live flow receipts packet from /flow_receipts"},
    {"command": "luci api prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]", "purpose": "show the live prompt ledger packets from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status"},
    {"command": "luci api prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]", "purpose": "show the raw prompt ledger rows from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status"},
    {"command": "luci api prompt recent [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_recent"},
    {"command": "luci api prompt filed [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompts_filed"},
    {"command": "luci api prompts filed [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompts_filed"},
    {"command": "luci api prompt links [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_work_order_links"},
    {"command": "luci api prompt work-order links [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_work_order_links"},
    {"command": "luci api prompt unlinked [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_unlinked"},
    {"command": "luci api prompt catalog [--json] [--base-url URL]", "purpose": "show the live prompt ledger summary packet from /prompt_catalog_status"},
    {"command": "luci api prompt catalog status [--json] [--base-url URL]", "purpose": "show the live prompt ledger summary packet from /prompt_catalog_status"},
    {"command": "luci api prompt raw recent [--json] [--base-url URL]", "purpose": "show the raw prompt recent rows from /prompt_recent"},
    {"command": "luci api prompt raw filed [--json] [--base-url URL]", "purpose": "show the raw prompt filed rows from /prompts_filed"},
    {"command": "luci api prompt raw links [--json] [--base-url URL]", "purpose": "show the raw prompt link rows from /prompt_work_order_links"},
    {"command": "luci api prompt raw unlinked [--json] [--base-url URL]", "purpose": "show the raw prompt unlinked rows from /prompt_unlinked"},
    {"command": "luci api prompt raw catalog [--json] [--base-url URL]", "purpose": "show the raw prompt catalog rows from /prompt_catalog_status"},
    {"command": "luci api book source [--json] [--base-url URL]", "purpose": "show the live book source packet from /book_source"},
    {"command": "luci api book scan [--json] [--base-url URL]", "purpose": "show the live book scan packet from /book_scan"},
    {"command": "luci api book read-queue [--json] [--base-url URL]", "purpose": "show the live book read queue packet from /book_read_queue"},
    {"command": "luci api book read queue [--json] [--base-url URL]", "purpose": "show the live book read queue packet from /book_read_queue"},
    {"command": "luci api book note [--json] [--base-url URL]", "purpose": "show the live book note packet from /book_note"},
    {"command": "luci api book candidate [--json] [--base-url URL]", "purpose": "show the live LoRA candidate packet from /lora_candidate"},
    {"command": "luci api book adapter [--json] [--base-url URL]", "purpose": "show the live LoRA adapter packet from /lora_adapter"},
    {"command": "luci api book training [--json] [--base-url URL]", "purpose": "show the live training packet from /training_job"},
    {"command": "luci api book receipt [--json] [--base-url URL]", "purpose": "show the live book receipt packet from /book_receipt"},
    {"command": "luci api book raw source [--json] [--base-url URL]", "purpose": "show the raw book source rows from /book_source"},
    {"command": "luci api book raw scan [--json] [--base-url URL]", "purpose": "show the raw book scan rows from /book_scan"},
    {"command": "luci api book raw read-queue [--json] [--base-url URL]", "purpose": "show the raw book read queue rows from /book_read_queue"},
    {"command": "luci api book raw note [--json] [--base-url URL]", "purpose": "show the raw book note rows from /book_note"},
    {"command": "luci api book raw candidate [--json] [--base-url URL]", "purpose": "show the raw book LoRA candidate rows from /lora_candidate"},
    {"command": "luci api book raw adapter [--json] [--base-url URL]", "purpose": "show the raw book LoRA adapter rows from /lora_adapter"},
    {"command": "luci api book raw training [--json] [--base-url URL]", "purpose": "show the raw book training rows from /training_job"},
    {"command": "luci api book raw receipt [--json] [--base-url URL]", "purpose": "show the raw book receipt rows from /book_receipt"},
    {"command": "luci api ontology work batch [--json] [--base-url URL]", "purpose": "show the live ontology work batch packet from /ontology_work_batch"},
    {"command": "luci api ontology work item [--json] [--base-url URL]", "purpose": "show the live ontology work item packet from /ontology_work_item"},
    {"command": "luci api ontology work raw batch [--json] [--base-url URL]", "purpose": "show the raw ontology work batch rows from /ontology_work_batch"},
    {"command": "luci api ontology work raw item [--json] [--base-url URL]", "purpose": "show the raw ontology work item rows from /ontology_work_item"},
    {"command": "luci api rpc subtree --root-id ID [--json] [--base-url URL]", "purpose": "show the live RPC subtree packet from /rpc/get_subtree"},
    {"command": "luci api rpc sort-key --node-id ID [--json] [--base-url URL]", "purpose": "show the live RPC sort-key packet from /rpc/fn_bible_node_sort_key"},
    {"command": "luci api rpc material --node-id ID [--json] [--base-url URL]", "purpose": "show the live RPC material packet from /rpc/fn_bible_node_material"},
    {"command": "luci api rpc cloud-packet --work-order-id ID [--json] [--base-url URL]", "purpose": "show the live cloud packet from /rpc/cloud_packet"},
    {"command": "luci api rpc file-prompt --payload-json JSON [--json] [--base-url URL]", "purpose": "show the live prompt-file packet from /rpc/file_prompt"},
    {"command": "luci api rpc decompose-prompt --payload-json JSON [--json] [--base-url URL]", "purpose": "show the live decompose packet from /rpc/decompose_prompt_to_work_orders"},
    {"command": "luci api rpc link-prompt --payload-json JSON [--json] [--base-url URL]", "purpose": "show the live prompt-link packet from /rpc/link_prompt_work_order"},
    {"command": "luci api cloud packet --work-order-id ID [--json] [--base-url URL]", "purpose": "request a bounded cloud packet from /rpc/cloud_packet"},
    {"command": "luci api cli process receipts [--json] [--base-url URL]", "purpose": "show the live CLI authority receipts from /cli_process_receipts"},
    {"command": "luci api payload archive status [--json] [--base-url URL]", "purpose": "show the cold payload archive status packet from /payload_archive_status"},
    {"command": "luci elastic shape <current|latest|residuals|pressure> [--json] [--base-url URL]", "purpose": "show the runtime elastic shape packets from /elastic_shape_current, /elastic_shape_latest, /shape_residuals_current, and /indy_attention_pressure_current"},
    {"command": "luci elastic shape emit --artifact-uuid UUID [--signal TOKEN=VALUE]... [--json] [--base-url URL]", "purpose": "emit and persist a runtime elastic shape receipt plus residuals"},
    {"command": "luci percyphon <current|matrix> [--json] [--base-url URL]", "purpose": "show the runtime Percyphon village packets from /percyphon_current and /percyphon_village_matrix"},
    {"command": "luci percyphon emit [--seed SEED] [--villager VALUE]... [--fluid-slots N] [--json] [--no-write-db]", "purpose": "emit a runtime Percyphon scaffold receipt and optionally write it to lucidota_go.percyphon_village"},
    {"command": "luci api bytewax windows [--json] [--base-url URL]", "purpose": "show the live Bytewax compact windows packet from /bytewax_compact_windows"},
    {"command": "luci api bytewax compact windows [--json] [--base-url URL]", "purpose": "show the live Bytewax compact windows packet from /bytewax_compact_windows"},
    {"command": "luci api bytewax raw windows [--json] [--base-url URL]", "purpose": "show the raw Bytewax compact windows rows from /bytewax_compact_windows"},
    {"command": "luci api book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]", "purpose": "show the live book / LoRA / training packets from book and training tables"},
    {"command": "luci api book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]", "purpose": "show the raw book / LoRA / training rows from book and training tables"},
    {"command": "luci api ontology work <batch|item> [--json] [--base-url URL]", "purpose": "show the live ontology work packets from /ontology_work_batch and /ontology_work_item"},
    {"command": "luci api ontology work raw <batch|item> [--json] [--base-url URL]", "purpose": "show the raw ontology work rows from /ontology_work_batch and /ontology_work_item"},
    {"command": "luci api test execution receipts [--json] [--base-url URL]", "purpose": "show the live API test execution receipt packet from /api_test_execution_receipts"},
    {"command": "luci api route catalog [--json] [--base-url URL]", "purpose": "show the live API route catalog packet from /api_route_catalog"},
    {"command": "luci api bible manuals [--json] [--base-url URL]", "purpose": "show the live bible manual packet from /api_bible_manuals"},
    {"command": "luci api bible route catalog [--json] [--base-url URL]", "purpose": "show the live bible route-catalog packet from /api_bible_route_catalog"},
    {"command": "luci api bible edges [--json] [--base-url URL]", "purpose": "show the live bible edge packet from /api_bible_edges"},
    {"command": "luci api bible nodes --manual-id ID [--json] [--base-url URL]", "purpose": "show the live bible node packet from /api_bible_nodes"},
    {"command": "luci api bible subtree --root-id ID [--json] [--base-url URL]", "purpose": "show the live bible subtree packet from /api_bible_subtree"},
    {"command": "luci rpc subtree --root-id ID [--json] [--base-url URL]", "purpose": "show the live canonical subtree packet from /rpc/get_subtree"},
    {"command": "luci rpc sort-key --node-id ID [--json] [--base-url URL]", "purpose": "show the live bible node sort-key packet from /rpc/fn_bible_node_sort_key"},
    {"command": "luci rpc material --node-id ID [--json] [--base-url URL]", "purpose": "show the live bible node material packet from /rpc/fn_bible_node_material"},
    {"command": "luci rpc cloud-packet --work-order-id ID [--json] [--base-url URL]", "purpose": "request a bounded cloud packet from /rpc/cloud_packet"},
    {"command": "luci rpc file-prompt --payload-json JSON [--json] [--base-url URL]", "purpose": "file a prompt directly through /rpc/file_prompt"},
    {"command": "luci rpc decompose-prompt --payload-json JSON [--json] [--base-url URL]", "purpose": "decompose a prompt directly through /rpc/decompose_prompt_to_work_orders"},
    {"command": "luci rpc link-prompt --payload-json JSON [--json] [--base-url URL]", "purpose": "link a prompt directly through /rpc/link_prompt_work_order"},
    {"command": "luci bible manuals [--json] [--base-url URL]", "purpose": "show the live bible manual packet from /api_bible_manuals"},
    {"command": "luci bible route catalog [--json] [--base-url URL]", "purpose": "show the live bible route-catalog packet from /api_bible_route_catalog"},
    {"command": "luci bible edges [--json] [--base-url URL]", "purpose": "show the live bible edge packet from /api_bible_edges"},
    {"command": "luci bible nodes --manual-id ID [--json] [--base-url URL]", "purpose": "show the live bible node packet from /api_bible_nodes"},
    {"command": "luci flow specs [--json] [--base-url URL]", "purpose": "show the live flow specs packet from /flow_specs"},
    {"command": "luci flow receipts [--json] [--base-url URL]", "purpose": "show the live flow receipts packet from /flow_receipts"},
    {"command": "luci daemon status [--json] [--base-url URL]", "purpose": "show the live daemon status packet from /daemon_status"},
    {"command": "luci prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]", "purpose": "show the live prompt ledger packets from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status"},
    {"command": "luci prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]", "purpose": "show the raw prompt ledger rows from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status"},
    {"command": "luci prompt recent [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_recent"},
    {"command": "luci prompt filed [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompts_filed"},
    {"command": "luci prompt links [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_work_order_links"},
    {"command": "luci prompt unlinked [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_unlinked"},
    {"command": "luci prompt catalog [--json] [--base-url URL]", "purpose": "show the live prompt ledger packet from /prompt_catalog_status"},
    {"command": "luci indy responses [--json] [--base-url URL]", "purpose": "show the live Indy response packet from /indy_responses"},
    {"command": "luci indy queue [--json] [--base-url URL]", "purpose": "show the live Indy queue packet from /indy_queue"},
    {"command": "luci bytewax windows [--json] [--base-url URL]", "purpose": "show the live Bytewax compact windows packet from /bytewax_compact_windows"},
    {"command": "luci bytewax raw windows [--json] [--base-url URL]", "purpose": "show the raw Bytewax compact windows rows from /bytewax_compact_windows"},
    {"command": "luci cloud packet --work-order-id ID [--json] [--base-url URL]", "purpose": "request a bounded cloud packet from /rpc/cloud_packet"},
    {"command": "luci book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]", "purpose": "show the live book / LoRA / training packets from book and training tables"},
    {"command": "luci book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]", "purpose": "show the raw book / LoRA / training rows from book and training tables"},
    {"command": "luci lora candidate [--json] [--base-url URL]", "purpose": "show the live LoRA candidate rows from /lora_candidate"},
    {"command": "luci lora adapter [--json] [--base-url URL]", "purpose": "show the live LoRA adapter rows from /lora_adapter"},
    {"command": "luci training job [--json] [--base-url URL]", "purpose": "show the live training rows from /training_job"},
    {"command": "luci ontology work <batch|item> [--json] [--base-url URL]", "purpose": "show the live ontology work packets from /ontology_work_batch and /ontology_work_item"},
    {"command": "luci ontology work raw <batch|item> [--json] [--base-url URL]", "purpose": "show the raw ontology work rows from /ontology_work_batch and /ontology_work_item"},
    {"command": "luci canon versions [--json] [--base-url URL]", "purpose": "show the live canon versions packet from /canon_versions"},
    {"command": "luci api canon versions [--json] [--base-url URL]", "purpose": "show the live canon versions packet from /canon_versions"},
    {"command": "luci cli-process-receipts [--json] [--base-url URL]", "purpose": "show the latest CLI authority receipts from /cli_process_receipts"},
    {"command": "luci cli-retention [--json] [--archive-all] [--max-rows N] [--older-than-hours N]", "purpose": "archive heavy CLI payload tails into cold storage"},
    {"command": "luci payload-archive-status [--json] [--base-url URL]", "purpose": "show the cold payload archive summary from /payload_archive_status"},
    {"command": "luci operate --text TEXT [--json]", "purpose": "normal operator chat/action path"},
    {"command": "luci /indy TEXT", "purpose": "operator chat routed through Indy conduit when invoked by chat text"},
    {"command": "luci indy-response [--json]", "purpose": "surface latest Indy response through /indy_responses"},
    {"command": "luci status", "purpose": "runtime status via LUCI engine"},
    {"command": "luci /flow | luci flow ui", "purpose": "open PromptFlow-style sidecar canvas"},
    {"command": "luci flow smoke [--json]", "purpose": "verify /flow sidecar wrapper"},
    {"command": "python3 scripts/prompt_ledger_capture.py [--decompose] [--json]", "purpose": "file steering prompts into /prompts_filed and optionally link work orders"},
    {"command": "luci sheet list --json", "purpose": "sheet/workflow surface"},
    {"command": "luci sheet current --json", "purpose": "show the live spreadsheet packet from /sheet_current"},
    {"command": "luci edge-loop-smoke --json", "purpose": "small operator edge smoke"},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_commands(api_routes: list[str]) -> list[dict[str, str]]:
    commands = list(BASE_COMMANDS)
    if "/root_law_docs" in api_routes:
        commands.append(
            {
                "command": "luci root-law-docs [--json] [--base-url URL]",
                "purpose": "show the live root-orchestrator manual from /root_law_docs",
            }
        )
        commands.append(
            {
                "command": "luci manual capsule [--json] [--base-url URL]",
                "purpose": "show the on-demand manual capsule from /root_law_docs",
            }
        )
    if "/root_orchestrator_current" in api_routes:
        commands.append(
            {
                "command": "luci root orchestrator current [--json] [--base-url URL]",
                "purpose": "show the live root orchestrator packet from /root_orchestrator_current",
            }
        )
    if "/manual_current" in api_routes:
        commands.append(
            {
                "command": "luci manual current [--json] [--base-url URL]",
                "purpose": "show the live manual packet from /manual_current",
            }
        )
    if any(route in api_routes for route in ("/percyphon_current", "/percyphon_village_matrix")):
        commands.append(
            {
                "command": "luci percyphon <current|matrix> [--json] [--base-url URL]",
                "purpose": "show the runtime Percyphon village packets from /percyphon_current and /percyphon_village_matrix",
            }
        )
    if "/capability_registry" in api_routes:
        commands.append(
            {
                "command": "luci capability list [--json] [--base-url URL]",
                "purpose": "show the live capability list surface",
            }
        )
    if "/active_goal" in api_routes:
        commands.append(
            {
                "command": "luci active goal [--json] [--base-url URL]",
                "purpose": "show the live active goal packet from /active_goal",
            }
        )
    if "/todo_current" in api_routes:
        commands.append(
            {
                "command": "luci todo current [--json] [--base-url URL]",
                "purpose": "show the live ontology todo batch from /todo_current",
            }
        )
    if "/canon_current" in api_routes:
        commands.append(
            {
                "command": "luci canon current [--json] [--base-url URL]",
                "purpose": "show the live canon packet from /canon_current",
            }
        )
        commands.append(
            {
                "command": "luci api canon current [--json] [--base-url URL]",
                "purpose": "show the live canon packet from /canon_current",
            }
        )
    if "/skill_policy_current" in api_routes:
        commands.append(
            {
                "command": "luci skill policy current [--json] [--base-url URL]",
                "purpose": "show the live skill policy packet from /skill_policy_current",
            }
        )
    if "/chrono_current" in api_routes:
        commands.append(
            {
                "command": "luci chrono current [--json] [--base-url URL]",
                "purpose": "show the live chrono packet from /chrono_current",
            }
        )
    if "/model_routing_current" in api_routes:
        commands.append(
            {
                "command": "luci model-routing-current [--json] [--base-url URL]",
                "purpose": "show the live model routing packet from /model_routing_current",
            }
        )
        commands.append(
            {
                "command": "luci model routing current [--json] [--base-url URL]",
                "purpose": "show the live model routing packet from /model_routing_current",
            }
        )
        commands.append(
            {
                "command": "luci model-routing-blockers [--json] [--base-url URL]",
                "purpose": "show only the missing-role blockers from /model_routing_current",
            }
        )
        commands.append(
            {
                "command": "luci model routing blockers [--json] [--base-url URL]",
                "purpose": "show only the missing-role blockers from /model_routing_current",
            }
        )
    if "/model_registry_current" in api_routes:
        commands.append(
            {
                "command": "luci model registry [--json] [--base-url URL]",
                "purpose": "show the raw model registry rows from /model_registry",
            }
        )
        commands.append(
            {
                "command": "luci model registry current [--json] [--base-url URL]",
                "purpose": "show the live model registry packet from /model_registry_current",
            }
        )
    if "/capability_current" in api_routes:
        commands.append(
            {
                "command": "luci capability current [--json] [--base-url URL]",
                "purpose": "show the live capability packet from /capability_current",
            }
        )
    if "/capability_registry" in api_routes:
        commands.append(
            {
                "command": "luci capability registry [--json] [--base-url URL]",
                "purpose": "show the live capability registry packet from /capability_registry",
            }
        )
        commands.append(
            {
                "command": "luci capability registry raw [--json] [--base-url URL]",
                "purpose": "show the raw capability registry rows from /capability_registry",
            }
        )
    if "/provider_current" in api_routes:
        commands.append(
            {
                "command": "luci provider current [--json] [--base-url URL]",
                "purpose": "show the live provider packet from /provider_current",
            }
        )
    if "/provider_registry" in api_routes:
        commands.append(
            {
                "command": "luci provider registry [--json] [--base-url URL]",
                "purpose": "show the live provider registry packet from /provider_registry",
            }
        )
        commands.append(
            {
                "command": "luci provider registry raw [--json] [--base-url URL]",
                "purpose": "show the raw provider registry rows from /provider_registry",
            }
        )
    if "/api_workflow_registry" in api_routes:
        commands.append(
            {
                "command": "luci workflow registry raw [--json] [--base-url URL]",
                "purpose": "show the raw workflow registry rows from /workflow_registry",
            }
        )
        commands.append(
            {
                "command": "luci workflow registry [--json] [--base-url URL]",
                "purpose": "show the live API workflow registry packet from /api_workflow_registry",
            }
        )
        commands.append(
            {
                "command": "luci api workflow registry [--json] [--base-url URL]",
                "purpose": "show the live API workflow registry packet from /api_workflow_registry",
            }
        )
        commands.append(
            {
                "command": "luci api workflow registry raw [--json] [--base-url URL]",
                "purpose": "show the raw API workflow registry rows from /api_workflow_registry",
            }
        )
    if "/api_route_catalog" in api_routes:
        commands.append(
            {
                "command": "luci api route catalog [--json] [--base-url URL]",
                "purpose": "show the live API route catalog packet from /api_route_catalog",
            }
        )
    if "/api_bible_manuals" in api_routes:
        commands.append(
            {
                "command": "luci bible manuals [--json] [--base-url URL]",
                "purpose": "show the live bible manual packet from /api_bible_manuals",
            }
        )
    if "/api_bible_route_catalog" in api_routes:
        commands.append(
            {
                "command": "luci bible route catalog [--json] [--base-url URL]",
                "purpose": "show the live bible route-catalog packet from /api_bible_route_catalog",
            }
        )
    if "/api_bible_edges" in api_routes:
        commands.append(
            {
                "command": "luci bible edges [--json] [--base-url URL]",
                "purpose": "show the live bible edge packet from /api_bible_edges",
            }
        )
    if "/api_bible_nodes" in api_routes:
        commands.append(
            {
                "command": "luci bible nodes --manual-id ID [--json] [--base-url URL]",
                "purpose": "show the live bible node packet from /api_bible_nodes",
            }
        )
    if "/api_bible_subtree" in api_routes:
        commands.append(
            {
                "command": "luci bible subtree --root-id ID [--json] [--base-url URL]",
                "purpose": "show the live bible subtree packet from /api_bible_subtree",
            }
        )
    if "/rpc/get_subtree" in api_routes:
        commands.append(
            {
                "command": "luci rpc subtree --root-id ID [--json] [--base-url URL]",
                "purpose": "show the live canonical subtree packet from /rpc/get_subtree",
            }
        )
    if "/rpc/fn_bible_node_sort_key" in api_routes:
        commands.append(
            {
                "command": "luci rpc sort-key --node-id ID [--json] [--base-url URL]",
                "purpose": "show the live bible node sort-key packet from /rpc/fn_bible_node_sort_key",
            }
        )
    if "/rpc/fn_bible_node_material" in api_routes:
        commands.append(
            {
                "command": "luci rpc material --node-id ID [--json] [--base-url URL]",
                "purpose": "show the live bible node material packet from /rpc/fn_bible_node_material",
            }
        )
    if "/rpc/cloud_packet" in api_routes:
        commands.append(
            {
                "command": "luci rpc cloud-packet --work-order-id ID [--json] [--base-url URL]",
                "purpose": "request a bounded cloud packet from /rpc/cloud_packet",
            }
        )
    if "/rpc/file_prompt" in api_routes:
        commands.append(
            {
                "command": "luci rpc file-prompt --payload-json JSON [--json] [--base-url URL]",
                "purpose": "file a prompt directly through /rpc/file_prompt",
            }
        )
    if "/rpc/decompose_prompt_to_work_orders" in api_routes:
        commands.append(
            {
                "command": "luci rpc decompose-prompt --payload-json JSON [--json] [--base-url URL]",
                "purpose": "decompose a prompt directly through /rpc/decompose_prompt_to_work_orders",
            }
        )
    if "/rpc/link_prompt_work_order" in api_routes:
        commands.append(
            {
                "command": "luci rpc link-prompt --payload-json JSON [--json] [--base-url URL]",
                "purpose": "link a prompt directly through /rpc/link_prompt_work_order",
            }
        )
    if "/flow_specs" in api_routes:
        commands.append(
            {
                "command": "luci flow specs [--json] [--base-url URL]",
                "purpose": "show the live flow specs packet from /flow_specs",
            }
        )
    if "/flow_receipts" in api_routes:
        commands.append(
            {
                "command": "luci flow receipts [--json] [--base-url URL]",
                "purpose": "show the live flow receipts packet from /flow_receipts",
            }
        )
    if "/daemon_status" in api_routes:
        commands.append(
            {
                "command": "luci daemon status [--json] [--base-url URL]",
                "purpose": "show the live daemon status packet from /daemon_status",
            }
        )
    if any(route in api_routes for route in ("/prompt_recent", "/prompts_filed", "/prompt_work_order_links", "/prompt_unlinked", "/prompt_catalog_status")):
        commands.append(
            {
                "command": "luci prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]",
                "purpose": "show the live prompt ledger packets from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status",
            }
        )
        commands.append(
            {
                "command": "luci prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]",
                "purpose": "show the raw prompt ledger rows from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status",
            }
        )
    if "/indy_queue" in api_routes:
        commands.append(
            {
                "command": "luci indy queue [--json] [--base-url URL]",
                "purpose": "show the live Indy queue packet from /indy_queue",
            }
        )
    if "/indy_responses" in api_routes:
        commands.append(
            {
                "command": "luci indy responses [--json] [--base-url URL]",
                "purpose": "show the live Indy response packet from /indy_responses",
            }
        )
    if "/bytewax_compact_windows" in api_routes:
        commands.append(
            {
                "command": "luci bytewax windows [--json] [--base-url URL]",
                "purpose": "show the live Bytewax compact windows packet from /bytewax_compact_windows",
            }
        )
        commands.append(
            {
                "command": "luci bytewax raw windows [--json] [--base-url URL]",
                "purpose": "show the raw Bytewax compact windows rows from /bytewax_compact_windows",
            }
        )
    if "/rpc/cloud_packet" in api_routes:
        commands.append(
            {
                "command": "luci cloud packet --work-order-id ID [--json] [--base-url URL]",
                "purpose": "request a bounded cloud packet from /rpc/cloud_packet",
            }
        )
    if "/api_root_law_docs" in api_routes:
        commands.append(
            {
                "command": "luci api root law docs [--json] [--base-url URL]",
                "purpose": "show the live root-law docs packet from /api_root_law_docs",
            }
        )
    if "/rpc/cloud_packet" in api_routes:
        commands.append(
            {
                "command": "luci api cloud packet --work-order-id ID [--json] [--base-url URL]",
                "purpose": "request a bounded cloud packet from /rpc/cloud_packet",
            }
        )
    if "/cli_process_receipts" in api_routes:
        commands.append(
            {
                "command": "luci api cli process receipts [--json] [--base-url URL]",
                "purpose": "show the live CLI authority receipts from /cli_process_receipts",
            }
        )
    if "/payload_archive_status" in api_routes:
        commands.append(
            {
                "command": "luci api payload archive status [--json] [--base-url URL]",
                "purpose": "show the cold payload archive status packet from /payload_archive_status",
            }
        )
    if "/bytewax_compact_windows" in api_routes:
        commands.append(
            {
                "command": "luci api bytewax windows [--json] [--base-url URL]",
                "purpose": "show the live Bytewax compact windows packet from /bytewax_compact_windows",
            }
        )
        commands.append(
            {
                "command": "luci api bytewax raw windows [--json] [--base-url URL]",
                "purpose": "show the raw Bytewax compact windows rows from /bytewax_compact_windows",
            }
        )
    if any(route in api_routes for route in ("/book_source", "/book_scan", "/book_read_queue", "/book_note", "/lora_candidate", "/lora_adapter", "/training_job", "/book_receipt")):
        commands.append(
            {
                "command": "luci api book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]",
                "purpose": "show the live book / LoRA / training packets from book and training tables",
            }
        )
        commands.append(
            {
                "command": "luci api book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]",
                "purpose": "show the raw book / LoRA / training rows from book and training tables",
            }
        )
    if any(route in api_routes for route in ("/ontology_work_batch", "/ontology_work_item")):
        commands.append(
            {
                "command": "luci api ontology work <batch|item> [--json] [--base-url URL]",
                "purpose": "show the live ontology work packets from /ontology_work_batch and /ontology_work_item",
            }
        )
        commands.append(
            {
                "command": "luci api ontology work raw <batch|item> [--json] [--base-url URL]",
                "purpose": "show the raw ontology work rows from /ontology_work_batch and /ontology_work_item",
            }
        )
    if any(route in api_routes for route in ("/prompt_recent", "/prompts_filed", "/prompt_work_order_links", "/prompt_unlinked", "/prompt_catalog_status")):
        commands.append(
            {
                "command": "luci api prompt <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]",
                "purpose": "show the live prompt ledger packets from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status",
            }
        )
        commands.append(
            {
                "command": "luci api prompt raw <recent|filed|links|unlinked|catalog> [--json] [--base-url URL]",
                "purpose": "show the raw prompt ledger rows from /prompt_recent, /prompts_filed, /prompt_work_order_links, /prompt_unlinked, /prompt_catalog_status",
            }
        )
        commands.append(
            {
                "command": "luci api prompt catalog [--json] [--base-url URL]",
                "purpose": "show the live prompt ledger summary packet from /prompt_catalog_status",
            }
        )
    if any(route in api_routes for route in ("/book_source", "/book_scan", "/book_read_queue", "/book_note", "/lora_candidate", "/lora_adapter", "/training_job", "/book_receipt")):
        commands.append(
            {
                "command": "luci book <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]",
                "purpose": "show the live book / LoRA / training packets from book and training tables",
            }
        )
        commands.append(
            {
                "command": "luci book raw <source|scan|read-queue|note|candidate|adapter|training|receipt> [--json] [--base-url URL]",
                "purpose": "show the raw book / LoRA / training rows from book and training tables",
            }
        )
        commands.append(
            {
                "command": "luci lora candidate [--json] [--base-url URL]",
                "purpose": "show the live LoRA candidate rows from /lora_candidate",
            }
        )
        commands.append(
            {
                "command": "luci lora adapter [--json] [--base-url URL]",
                "purpose": "show the live LoRA adapter rows from /lora_adapter",
            }
        )
        commands.append(
            {
                "command": "luci training job [--json] [--base-url URL]",
                "purpose": "show the live training rows from /training_job",
            }
        )
    if any(route in api_routes for route in ("/ontology_work_batch", "/ontology_work_item")):
        commands.append(
            {
                "command": "luci ontology work <batch|item> [--json] [--base-url URL]",
                "purpose": "show the live ontology work packets from /ontology_work_batch and /ontology_work_item",
            }
        )
        commands.append(
            {
                "command": "luci ontology work raw <batch|item> [--json] [--base-url URL]",
                "purpose": "show the raw ontology work rows from /ontology_work_batch and /ontology_work_item",
            }
        )
    if "/canon_versions" in api_routes:
        commands.append(
            {
                "command": "luci canon versions [--json] [--base-url URL]",
                "purpose": "show the live canon versions packet from /canon_versions",
            }
        )
        commands.append(
            {
                "command": "luci api canon versions [--json] [--base-url URL]",
                "purpose": "show the live canon versions packet from /canon_versions",
            }
        )
    if "/sheet_current" in api_routes:
        commands.append(
            {
                "command": "luci sheet current --json",
                "purpose": "show the live spreadsheet packet from /sheet_current",
            }
        )
    if "/workflow_current" in api_routes:
        commands.append(
            {
                "command": "luci workflow current [--json] [--base-url URL]",
                "purpose": "show the live workflow packet from /workflow_current",
            }
        )
    return commands


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> tuple[bool, Any, str]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return True, json.loads(resp.read().decode("utf-8") or "null"), url
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        return False, {"error": f"HTTPError:{exc.code}", "body": body}, url
    except Exception as exc:
        return False, {"error": f"{type(exc).__name__}: {exc}"}, url


def dedupe_refs(*groups: list[Any]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for ref in group:
            ref_text = str(ref)
            if ref_text and ref_text not in seen:
                seen.add(ref_text)
                refs.append(ref_text)
    return refs


def build_payload(mode: str, base_url: str) -> dict[str, Any]:
    api_ok, openapi, openapi_url = fetch_json(base_url, "")
    manual_ok, manuals, manual_url = fetch_json(base_url, "manual_current", {"order": "manual_id.asc"})
    root_ok, root_rows, root_url = fetch_json(
        base_url,
        "root_orchestrator_current",
        {"select": "orchestrator_id,title,route_list,next_command_refs,orchestration,live_surface", "limit": "1"},
    )
    cli_ok, cli_rows, cli_url = fetch_json(
        base_url,
        "cli_process_receipts",
        {"select": "*,next_command_refs", "order": "received_at.desc", "limit": "1"},
    )
    payload_ok, payload_rows, payload_url = fetch_json(base_url, "payload_archive_status", {"order": "source_table.asc,payload_kind.asc", "limit": "20"})
    paths = sorted((openapi or {}).get("paths", {}).keys()) if api_ok and isinstance(openapi, dict) else []
    manual_rows = manuals if manual_ok and isinstance(manuals, list) else []
    manual_row = manual_rows[0] if manual_rows and isinstance(manual_rows[0], dict) else {}
    root_rows_list = root_rows if root_ok and isinstance(root_rows, list) else []
    root_row = root_rows_list[0] if root_rows_list and isinstance(root_rows_list[0], dict) else {}
    summary_row = manual_row or root_row
    summary_source = "manual_current" if manual_row else ("root_orchestrator_current" if root_row else "")
    summary_live_surface = summary_row.get("live_surface") if isinstance(summary_row.get("live_surface"), dict) else {}
    cli_receipts = cli_rows if cli_ok and isinstance(cli_rows, list) else []
    payload_archives = payload_rows if payload_ok and isinstance(payload_rows, list) else []
    route_refs = summary_row.get("route_refs") or [
        str(route.get("route_id"))
        for route in (summary_row.get("route_list") or [])
        if isinstance(route, dict) and route.get("route_id")
    ]
    surface_refs = summary_row.get("surface_refs") or list(route_refs)
    renderer_refs = summary_row.get("renderer_refs") or ["renderer_registry", "command_registry"]
    capability_refs = summary_row.get("capability_refs") or [
        str(cap.get("capability_key"))
        for cap_packet in (summary_live_surface.get("capability_current") or [])
        if isinstance(cap_packet, dict)
        for cap in (cap_packet.get("active_capabilities") or [])
        if isinstance(cap, dict) and cap.get("capability_key")
    ]
    next_command_refs = dedupe_refs(
        summary_row.get("next_command_refs") or [],
        [*route_refs, *surface_refs, *renderer_refs, *capability_refs],
    )
    return {
        "schema": "lucidota.luci.help_manual.v1",
        "generated_at": now(),
        "mode": mode,
        "source": "postgrest_safe_surface",
        "postgrest_base_url": base_url.rstrip("/"),
        "api_status": "ok" if api_ok and manual_ok else "degraded",
        "openapi_url": openapi_url,
        "manual_current_url": manual_url,
        "root_orchestrator_current_url": root_url,
        "manuals": manual_rows,
        "manual_summary": {
            "summary_source": summary_source,
            "manual_id": summary_row.get("manual_id"),
            "orchestrator_id": summary_row.get("orchestrator_id"),
            "title": summary_row.get("title"),
            "node_count": summary_row.get("node_count"),
            "route_count": summary_row.get("route_count"),
            "max_updated_at": summary_row.get("max_updated_at"),
            "route_refs": route_refs,
            "surface_refs": surface_refs,
            "renderer_refs": renderer_refs,
            "capability_refs": capability_refs,
            "next_command_refs": next_command_refs,
        },
        "api_routes": paths,
        "cli_process_receipts_url": cli_url,
        "cli_process_receipts": cli_receipts,
        "payload_archive_status_url": payload_url,
        "payload_archive_status": payload_archives,
        "next_command_refs": next_command_refs,
        "route_refs": route_refs,
        "surface_refs": surface_refs,
        "renderer_refs": renderer_refs,
        "capability_refs": capability_refs,
        "commands": build_commands(paths),
        "forbidden_detours": ["Phantom takeover", "Root-Rotor detour", "loose JSON queue authority"],
    }


def render_text(payload: dict[str, Any]) -> str:
    title = "LUCI MANUAL" if payload["mode"] == "manual" else "LUCI HELP"
    lines = [
        title,
        f"API_STATUS={payload['api_status']}",
        f"POSTGREST={payload['postgrest_base_url']}",
        "",
        "Commands:",
    ]
    for row in payload["commands"]:
        lines.append(f"- {row['command']} :: {row['purpose']}")
    lines.extend(["", "Manual volumes from /manual_current:"])
    if payload["manuals"]:
        summary = payload.get("manual_summary") if isinstance(payload.get("manual_summary"), dict) else {}
        summary_label = summary.get("manual_id") or summary.get("orchestrator_id")
        lines.append(
            f"- {summary_label} :: {summary.get('title')} nodes={summary.get('node_count') or summary.get('route_count')} updated={summary.get('max_updated_at')}"
        )
        row = payload["manuals"][0]
        route_list = row.get("route_list") if isinstance(row, dict) else None
        if isinstance(route_list, list) and route_list:
            lines.append("  Routes:")
            for route in route_list[:24]:
                lines.append(
                    f"  - {route.get('method')} {route.get('path_pattern')} :: {route.get('description')} "
                    f"[{route.get('status')}]"
                )
        route_refs = summary.get("route_refs") if isinstance(summary, dict) else None
        if isinstance(route_refs, list) and route_refs:
            lines.append(f"  Route refs: {', '.join(str(ref) for ref in route_refs[:16])}")
        capability_refs = summary.get("capability_refs") if isinstance(summary, dict) else None
        if isinstance(capability_refs, list) and capability_refs:
            lines.append(f"  Capability refs: {', '.join(str(ref) for ref in capability_refs[:16])}")
        surface_refs = summary.get("surface_refs") if isinstance(summary, dict) else None
        if isinstance(surface_refs, list) and surface_refs:
            lines.append(f"  Surface refs: {', '.join(str(ref) for ref in surface_refs[:16])}")
        renderer_refs = summary.get("renderer_refs") if isinstance(summary, dict) else None
        if isinstance(renderer_refs, list) and renderer_refs:
            lines.append(f"  Renderer refs: {', '.join(str(ref) for ref in renderer_refs[:16])}")
        next_command_refs = summary.get("next_command_refs") if isinstance(summary, dict) else None
        if isinstance(next_command_refs, list) and next_command_refs:
            lines.append(f"  Next command refs: {', '.join(str(ref) for ref in next_command_refs[:16])}")
        live_surface = row.get("live_surface") if isinstance(row, dict) else None
        if isinstance(live_surface, dict):
            bible_nodes = live_surface.get("api_bible_nodes") or []
            if isinstance(bible_nodes, list) and bible_nodes:
                lines.append(f"  Bible nodes: {len(bible_nodes)} rows (manual {row.get('manual_id')})")
            canon_versions = live_surface.get("canon_versions") or []
            if isinstance(canon_versions, list) and canon_versions:
                lines.append(f"  Canon versions: {len(canon_versions)} rows")
            bible_manuals = live_surface.get("api_bible_manuals") or []
            if isinstance(bible_manuals, list) and bible_manuals:
                lines.append(f"  Bible manuals: {len(bible_manuals)} rows")
            bible_route_catalog = live_surface.get("api_bible_route_catalog") or []
            if isinstance(bible_route_catalog, list) and bible_route_catalog:
                lines.append(f"  Bible route catalog: {len(bible_route_catalog)} rows")
            bible_edges = live_surface.get("api_bible_edges") or []
            if isinstance(bible_edges, list) and bible_edges:
                lines.append(f"  Bible edges: {len(bible_edges)} rows")
            current_goal = live_surface.get("current_goal") or {}
            if isinstance(current_goal, dict) and current_goal:
                lines.append(f"  Current goal: {current_goal.get('title')} ({current_goal.get('status')})")
            daemon_status = live_surface.get("daemon_status") or []
            if isinstance(daemon_status, list) and daemon_status:
                lines.append(f"  Daemon count: {len(daemon_status)}")
            todo_current = live_surface.get("todo_current") or []
            if isinstance(todo_current, list) and todo_current:
                top_todo = todo_current[0]
                lines.append(
                    f"  Todo current: {top_todo.get('batch_key')} :: {top_todo.get('objective_summary')} "
                    f"[items={top_todo.get('item_count')}]"
                )
        next_commands = row.get("next_commands") if isinstance(row, dict) else None
        if isinstance(next_commands, list) and next_commands:
            lines.append("  Next commands:")
            for cmd in next_commands[:8]:
                lines.append(f"  - {cmd}")
        packet_next_command_refs = row.get("next_command_refs") if isinstance(row, dict) else None
        if isinstance(packet_next_command_refs, list) and packet_next_command_refs:
            lines.append(
                "  Packet next command refs: "
                + ", ".join(str(ref) for ref in packet_next_command_refs[:16])
            )
    cli_rows = payload.get("cli_process_receipts") if isinstance(payload, dict) else None
    if isinstance(cli_rows, list) and cli_rows:
        row = cli_rows[0]
        lines.append("")
        lines.append("Latest CLI receipts from /cli_process_receipts:")
        lines.append(
            f"- {row.get('receipt_uuid')} :: {row.get('status')} pid={row.get('process_pid')} "
            f"restarts={row.get('restart_count')} auth_injected={row.get('auth_injected')}"
        )
        if row.get("command_line"):
            lines.append(f"  command: {row.get('command_line')}")
    else:
        lines.append("")
        lines.append("Latest CLI receipts from /cli_process_receipts:")
        lines.append("- unavailable: check /cli_process_receipts and PostgREST readiness")
    archive_rows = payload.get("payload_archive_status") if isinstance(payload, dict) else None
    if isinstance(archive_rows, list) and archive_rows:
        lines.append("")
        lines.append("Cold payload archive status from /payload_archive_status:")
        for row in archive_rows[:6]:
            lines.append(
                f"- {row.get('source_table')}::{row.get('payload_kind')} count={row.get('archive_count')} "
                f"bytes={row.get('archived_bytes')} latest={row.get('latest_archived_at')}"
            )
    else:
        lines.append("")
        lines.append("Cold payload archive status from /payload_archive_status:")
        lines.append("- unavailable: check /payload_archive_status and PostgREST readiness")
    lines.extend(["", f"OpenAPI routes visible: {len(payload['api_routes'])}"])
    for route in payload["api_routes"][:40]:
        lines.append(f"- {route}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="LUCI help/manual from PostgREST safe surfaces.")
    parser.add_argument("mode", nargs="?", choices=["help", "manual"], default="help")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_payload(args.mode, args.base_url)
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(render_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
