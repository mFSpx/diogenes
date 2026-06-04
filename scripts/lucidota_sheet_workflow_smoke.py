#!/usr/bin/env python3
"""Emit a no-DB receipt for sheet-first workflow spine routing.

Used as a dry-run proof: no DB connections, no model calls, no external IO
beyond manifest/schema files. Shows where ingest/evidence/graph/forms/network
analysis domains sit in SQL/sheet-first route order.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_WORKFLOW_MANIFEST = ROOT / "04_RUNTIME/lucidota_workflow_registry.json"
DEFAULT_SHEET_MANIFEST = ROOT / "04_RUNTIME/lucidota_sheet_manifest.json"
DEFAULT_SCHEMA_FILE = ROOT / "06_SCHEMA/052_lucidota_sheet_workflow_layer.sql"
DEFAULT_POSTGREST_CONF = ROOT / "GOALS/root_rotor_postgrest.conf"
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS/runtime/lucidota_sheet_workflow_smoke_latest.json"

REQUIRED_DOMAINS = [
    {
        "logical": "ingest",
        "registry_domain": "korpus_ingest",
        "target": "workflow.korpus_ingest.route",
        "route_stage": "live_view",
    },
    {
        "logical": "evidence_ingest",
        "registry_domain": "evidence_ingest",
        "target": "workflow.evidence_ingest.capture",
        "route_stage": "import_sheet",
    },
    {
        "logical": "graph_ops",
        "registry_domain": "graph_ops",
        "target": "workflow.graph_ops.materialize",
        "route_stage": "promotion_sheet",
    },
    {
        "logical": "documents_forms",
        "registry_domain": "documents_forms",
        "target": "workflow.documents_forms.packetize",
        "route_stage": "export_sheet",
    },
    {
        "logical": "network_analysis",
        "registry_domain": "network_analysis",
        "target": "workflow.network_analysis.centrality",
        "route_stage": "pivot_sheet",
    },
]

SQL_CONTRACT_SNIPPETS = [
    "CREATE SCHEMA IF NOT EXISTS lucidota_sheet;",
    "CREATE SCHEMA IF NOT EXISTS lucidota_scratch;",
    "CREATE SCHEMA IF NOT EXISTS lucidota_projection;",
    "CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_workflow_route",
    "CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_workflow_receipt",
    "CREATE UNLOGGED TABLE IF NOT EXISTS lucidota_scratch.sheet_workflow_route_scratch",
    "CREATE OR REPLACE VIEW lucidota_sheet.sheet_workflow_head",
    "CREATE OR REPLACE VIEW lucidota_projection.sheet_workflow_route_sheet",
    "CREATE MATERIALIZED VIEW IF NOT EXISTS lucidota_projection.workflow_domain_pressure_sheet",
    "CREATE OR REPLACE FUNCTION lucidota_sheet.record_sheet_workflow_receipt",
]


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_query_sql(route: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    query = str(route.get("query_sql") or "")
    upper = query.upper()
    if not query:
        errors.append("query_sql_missing")
        return errors
    if "SELECT *" in upper:
        errors.append("select_star_forbidden")
    if "LIMIT" not in upper and "REFRESH MATERIALIZED VIEW" not in upper:
        errors.append("live_query_without_limit")
    return errors


def validate_schema_contract(text: str) -> tuple[bool, list[str]]:
    missing = [snippet for snippet in SQL_CONTRACT_SNIPPETS if snippet not in text]
    return not missing, missing


def parse_postgrest_conf(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = [piece.strip() for piece in line.split("=", 1)]
        entries[key] = value.strip().strip('"')
    return entries


def build_receipt(
    workflow_path: Path,
    sheet_path: Path,
    schema_path: Path,
    postgrest_conf: Path,
) -> dict[str, Any]:
    workflow = load_json(workflow_path)
    sheet = load_json(sheet_path)
    workflows = workflow.get("workflows", [])
    route_by_target = {item.get("target"): item for item in workflows}
    route_by_domain = {item.get("domain"): item for item in workflows}

    schema_text = schema_path.read_text(encoding="utf-8")
    schema_ok, missing_snippets = validate_schema_contract(schema_text)

    route_items: list[dict[str, Any]] = []
    errors: list[str] = []
    if not schema_ok:
        errors.append(f"schema_contract_missing:{','.join(missing_snippets)}")

    for order, route in enumerate(REQUIRED_DOMAINS, start=1):
        selected = route_by_target.get(route["target"]) or route_by_domain.get(route["registry_domain"])
        if not selected:
            errors.append(f"missing_route:{route['logical']}")
            continue

        q_errors = validate_query_sql(selected)
        for q_error in q_errors:
            errors.append(f"{route['logical']}:{q_error}")
        if not selected.get("receipt_required", True):
            errors.append(f"{route['logical']}:receipt_not_required")

        query_sql = str(selected.get("query_sql") or "")
        route_items.append(
            {
                "sequence": order,
                "logical_domain": route["logical"],
                "registry_domain": route["registry_domain"],
                "target": selected.get("target"),
                "task_class": selected.get("task_class"),
                "route_stage": route["route_stage"],
                "query_hash": sha256(query_sql.encode("utf-8")).hexdigest(),
                "sql_preview": " ".join(query_sql.split()),
                "query_sql": query_sql,
                "query_has_limit": ("LIMIT" in query_sql.upper())
                or ("REFRESH MATERIALIZED VIEW" in query_sql.upper()),
                "sheet_routing_alignment": [
                    step for step in sheet.get("routing_order", []) if step in [
                        "generated_column",
                        "live_view",
                        "materialized_projection",
                        "sql_aggregate",
                        "duckdb_file_sheet",
                        "algorithm_escalation",
                        "model_last_resort",
                    ]
                ],
            }
        )

    postgrest = parse_postgrest_conf(postgrest_conf)
    sheet_routing_order = sheet.get("routing_order", [])
    workflow_order = workflow.get("routing_law", [])

    receipt: dict[str, Any] = {
        "schema": "lucidota.sheet_workflow_smoke_receipt.v1",
        "generated_at": now_z(),
        "status": "PASS",
        "execution": "dry_run",
        "db_connected": False,
        "db_mode": "no_db",
        "postgrest_conf": rel(postgrest_conf),
        "postgrest": {
            "enabled": bool(postgrest),
            "schemas": postgrest.get("db-schemas", "").split(",") if postgrest else [],
            "server_host": postgrest.get("server-host"),
            "server_port": postgrest.get("server-port"),
            "anon_role": postgrest.get("db-anon-role"),
        },
        "schema_file": rel(schema_path),
        "sheet_first_routing_order": sheet_routing_order,
        "workflow_routing_law": workflow_order,
        "route_order": [item["logical_domain"] for item in route_items],
        "routes": route_items,
        "required_domains": [r["logical"] for r in REQUIRED_DOMAINS],
        "errors": errors,
    }

    if errors:
        receipt["status"] = "FAIL"

    digest_payload = {
        k: receipt[k]
        for k in receipt
        if k not in {"output_hash"}
    }
    receipt["output_hash"] = sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luci-sheet-workflow-smoke")
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW_MANIFEST))
    parser.add_argument("--sheet-manifest", default=str(DEFAULT_SHEET_MANIFEST))
    parser.add_argument("--schema-file", default=str(DEFAULT_SCHEMA_FILE))
    parser.add_argument("--postgrest-conf", default=str(DEFAULT_POSTGREST_CONF))
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--json", action="store_true", default=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    sheet_path = Path(args.sheet_manifest)
    schema_path = Path(args.schema_file)
    postgrest_conf = Path(args.postgrest_conf)

    for candidate in [workflow_path, sheet_path, schema_path]:
        if not candidate.exists():
            raise SystemExit(f"INPUT_MISSING:{candidate}")

    receipt = build_receipt(
        workflow_path=workflow_path,
        sheet_path=sheet_path,
        schema_path=schema_path,
        postgrest_conf=postgrest_conf,
    )

    out = Path(args.receipt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(receipt, sort_keys=True) if args.json else str(receipt))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
