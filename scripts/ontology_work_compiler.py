#!/usr/bin/env python3
"""Compile messy operator text into DB-visible ontology work batches.

The compiler is deterministic and bounded:
- split operator text into sections/items,
- label each item with subsystem, GO/CO/IO-ish tags, risk, parallelism, and
  functionality-preservation contract,
- discover live model/provider lanes from PostgREST,
- persist the batch/items into lucidota_control tables,
- expose the active todo batch through lucidota_canon.todo_current.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import indy_runtime_broker

DB_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
MAX_SOURCE_EXCERPT = 12000
ROLE_ORDER = ("router", "classifier", "summarizer", "embedder", "reranker", "thinker", "watcher", "treelite_gate")


def sql_placeholder_count(sql: str) -> int:
    return sql.count("%s")


def execute_with_bind_guard(cur: Any, sql: str, params: tuple[Any, ...] | list[Any]) -> None:
    placeholder_count = sql_placeholder_count(sql)
    bind_count = len(params)
    if placeholder_count != bind_count:
        raise ValueError(f"sql_bind_mismatch placeholder_count={placeholder_count} bind_count={bind_count}")
    cur.execute(sql, params)


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def normalize_text(text: str) -> str:
    return re.sub(r"\r\n?", "\n", text).strip()


def split_sections(text: str) -> list[str]:
    lines = normalize_text(text).splitlines()
    if not lines:
        return []

    sections: list[list[str]] = []
    current: list[str] = []
    numbered = re.compile(r"^\s*(\d+)[\).\:-]\s+")

    for line in lines:
        if numbered.match(line):
            if current:
                sections.append(current)
            current = [line]
            continue
        if current:
            current.append(line)
        elif line.strip():
            current = [line]

    if current:
        sections.append(current)

    if not sections:
        return [normalize_text(text)]

    return [normalize_text("\n".join(section)) for section in sections if any(part.strip() for part in section)]


def section_title(section_text: str) -> str:
    first_line = section_text.splitlines()[0].strip()
    first_line = re.sub(r"^\s*\d+[).\:-]\s*", "", first_line)
    return first_line


def infer_subsystem(title: str, body: str) -> str:
    low = f"{title}\n{body}".lower()
    if any(token in low for token in ("manual", "api", "route", "postgrest", "cloud_packet", "canon")):
        return "manual_api"
    if any(token in low for token in ("book", "books", "lora", "adapter", "training")):
        return "book_ops"
    if any(token in low for token in ("indy daemon", "daemon", "front door", "queue", "response")):
        return "indy_daemon"
    if any(token in low for token in ("model", "provider", "workflow", "needle", "treelite", "router", "classifier", "summarizer", "embedder", "reranker", "thinker", "watcher")):
        return "model_orchestration"
    if any(token in low for token in ("rust", "rewrite", "port")):
        return "rust_rewrite"
    if any(token in low for token in ("ingest", "canon", "artifact", "legacy", "corpse", "duplicate", "review")):
        return "ingest_and_slop"
    if any(token in low for token in ("test", "proof", "verify", "receipt", "smoke")):
        return "verification"
    return "mixed"


def infer_work_kind(title: str, body: str) -> str:
    low = f"{title}\n{body}".lower()
    if any(token in low for token in ("workflow", "workflow_registry", "basic workflow", "basic-workflows")):
        return "workflow"
    if any(token in low for token in ("manual", "api", "route", "postgrest", "route audit", "readable through postgrest")):
        return "audit"
    if any(token in low for token in ("retire", "migrate", "schema", "db visible", "route planner", "compiler")):
        return "migration"
    if any(token in low for token in ("daemon", "front door", "service", "loop", "once", "queue")):
        return "service"
    if any(token in low for token in ("test", "proof", "receipt", "verify", "e2e", "smoke")):
        return "test"
    if any(token in low for token in ("model", "provider", "workflow", "needle", "treelite", "router", "classifier", "summarizer", "embedder", "reranker", "thinker", "watcher")):
        return "routing"
    if any(token in low for token in ("rust", "rewrite", "port")):
        return "rewrite"
    if any(token in low for token in ("quarantine", "slop", "duplicate", "corpse", "legacy")):
        return "cleanup"
    if any(token in low for token in ("ingest", "classify", "promote")):
        return "ingest"
    return "audit"


def infer_risk(title: str, body: str) -> str:
    low = f"{title}\n{body}".lower()
    if any(token in low for token in ("rust", "rewrite", "delete", "remove", "retire")):
        return "destructive"
    if any(token in low for token in ("migrate", "service", "daemon", "schema", "db", "port")):
        return "high"
    if any(token in low for token in ("test", "verify", "audit", "read", "inspect")):
        return "low"
    return "medium"


def infer_tags(title: str, body: str) -> list[str]:
    low = f"{title}\n{body}".lower()
    tags: list[str] = []
    mapping = [
        ("manual", "MANUAL"),
        ("api", "API"),
        ("route", "ROUTE"),
        ("postgrest", "POSTGREST"),
        ("book", "BOOK"),
        ("books", "BOOK"),
        ("lora", "LORA"),
        ("adapter", "ADAPTER"),
        ("training", "TRAINING"),
        ("daemon", "DAEMON"),
        ("queue", "QUEUE"),
        ("response", "RESPONSE"),
        ("model", "MODEL"),
        ("provider", "PROVIDER"),
        ("workflow", "WORKFLOW"),
        ("needle", "NEEDLE"),
        ("treelite", "TREELITE"),
        ("rust", "RUST"),
        ("ingest", "INGEST"),
        ("receipt", "RECEIPT"),
        ("test", "TEST"),
        ("proof", "PROOF"),
        ("parallel", "PARALLEL"),
        ("serialize", "SERIAL"),
        ("quarantine", "QUARANTINE"),
        ("slop", "SLOP"),
        ("legacy", "LEGACY"),
    ]
    for needle, tag in mapping:
        if needle in low and tag not in tags:
            tags.append(tag)
    if not tags:
        tags = ["OBJECT", "WORK"]
    return tags[:8]


def infer_parallelizable(title: str, body: str) -> tuple[bool, bool, str]:
    low = f"{title}\n{body}".lower()
    if any(token in low for token in ("db migration", "shared core", "service", "daemon", "rewrite", "delete", "retire")):
        return False, True, "serialized"
    if any(token in low for token in ("test", "audit", "read only", "classification", "discover", "inspect")):
        return True, False, "parallel"
    return True, False, "mixed"


def inference_target_role(kind: str, title: str, body: str) -> str:
    low = f"{title}\n{body}".lower()
    if kind in {"audit", "routing"}:
        return "router"
    if kind in {"test"}:
        return "treelite_gate"
    if kind in {"service", "rewrite", "migration"}:
        return "thinker"
    if kind in {"ingest", "cleanup"}:
        return "watcher"
    if any(token in low for token in ("classify", "classifier")):
        return "classifier"
    return "router"


def build_acceptance_test(kind: str, title: str) -> str:
    if kind == "workflow":
        return "query workflow_registry, verify the workflow remains active, and show the batch in manual_current/todo_current."
    if kind == "audit":
        return "curl the live route and verify the route list/manual packet reflects the current API truth."
    if kind == "migration":
        return "apply the schema or table change, reload PostgREST, and verify the new rows are readable."
    if kind == "service":
        return "restart the live service and verify daemon_status shows the DB-backed front door."
    if kind == "test":
        return "run receipt-gated targeted tests and verify the new route/manual assertions pass."
    if kind == "routing":
        return "query model_registry/provider_registry, select actual live roles, and expose missing roles as blockers."
    if kind == "rewrite":
        return "port one bounded module, run A/B equivalence, and keep the API surface stable."
    if kind == "cleanup":
        return "quarantine by hash/receipt first; delete only after tests prove no live dependency."
    return f"prove: {title}"


def build_receipt_requirement(kind: str) -> str:
    if kind in {"audit", "test", "workflow"}:
        return "receipt-gated verification required"
    if kind in {"migration", "service", "rewrite"}:
        return "receipt + live API readback required"
    return "receipt-backed row or quarantine receipt required"


def build_functionality_contract(kind: str) -> str:
    contracts = {
        "audit": "Do not regress live API/manual truth while adding route visibility.",
        "migration": "Preserve existing DB behavior and expose new rows through PostgREST.",
        "service": "Keep the daemon DB-driven; no BOOKS-folder authority.",
        "test": "Do not fake the gate; prove the live path.",
        "workflow": "Preserve existing simple workflows as workflow rows; do not flatten them into ad hoc script notes.",
        "routing": "Discover real local lanes only; expose missing roles as blockers.",
        "rewrite": "Keep API surface stable and retire old code only after equivalence receipts.",
        "cleanup": "Quarantine before deletion; preserve functionality, not nostalgia.",
    }
    return contracts.get(kind, "Preserve the current functionality contract.")


def planner_group(kind: str, parallelizable: bool) -> str:
    if not parallelizable:
        return "serialized_core"
    if kind in {"audit", "test", "routing", "cleanup"}:
        return "parallel_scan"
    return "mixed_queue"


def route_hint(subsystem: str, kind: str) -> str:
    if subsystem == "manual_api":
        return "/manual_current"
    if subsystem == "book_ops":
        return "/book_read_queue"
    if subsystem == "indy_daemon":
        return "/indy_queue"
    if kind == "workflow":
        return "/workflow_registry"
    if subsystem == "model_orchestration":
        return "/model_registry"
    if subsystem == "verification":
        return "/todo_current"
    if kind == "rewrite":
        return "scripts/rust_port_candidate"
    return "/todo_current"


def lane_summary(required_role: str, base_url: str) -> dict[str, Any]:
    model = indy_runtime_broker.choose_local_model(role=required_role, base_url=base_url)
    if model:
        return {
            "required_role": required_role,
            "selected_role": model.get("role", required_role),
            "selected_model_id": model.get("model_id", ""),
            "provider_key": "local_model",
            "status": "ready",
            "reason": "live local lane available",
        }
    router = indy_runtime_broker.choose_local_model(role="router", base_url=base_url)
    if router:
        return {
            "required_role": required_role,
            "selected_role": router.get("role", "router"),
            "selected_model_id": router.get("model_id", ""),
            "provider_key": "local_model",
            "status": "degraded",
            "reason": f"required role missing; fell back to router lane",
        }
    return {
        "required_role": required_role,
        "selected_role": "",
        "selected_model_id": "",
        "provider_key": "",
        "status": "blocked",
        "reason": f"required role {required_role!r} not present in model_registry",
    }


def discover_lanes(base_url: str) -> tuple[list[dict[str, Any]], list[str]]:
    snapshot = indy_runtime_broker.registry_snapshot(base_url=base_url)
    role_rows = snapshot.get("local_model_roles") if isinstance(snapshot, dict) else {}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    if isinstance(role_rows, dict):
        for role in ROLE_ORDER:
            row = role_rows.get(role)
            if row:
                selected.append(
                    {
                        "role": role,
                        "model_id": row.get("model_id", ""),
                        "slot_name": row.get("slot_name", ""),
                        "status": "ready",
                    }
                )
            else:
                missing.append(role)
    else:
        missing = list(ROLE_ORDER)
    return selected, missing


def compile_sections(operator_text: str, *, base_url: str = indy_runtime_broker.DEFAULT_BASE_URL, source_ref: str = "operator_turn") -> dict[str, Any]:
    text = normalize_text(operator_text)
    sections = split_sections(text)
    selected_lanes, missing_roles = discover_lanes(base_url)
    items: list[dict[str, Any]] = []
    dependency_edges: list[dict[str, Any]] = []

    for idx, section in enumerate(sections, 1):
        title = section_title(section)
        subsystem = infer_subsystem(title, section)
        work_kind = infer_work_kind(title, section)
        risk = infer_risk(title, section)
        parallelizable, serialized, parallel_policy = infer_parallelizable(title, section)
        tags = infer_tags(title, section)
        required_role = inference_target_role(work_kind, title, section)
        lane = lane_summary(required_role, base_url)
        item = {
            "item_rank": idx,
            "planner_group": planner_group(work_kind, parallelizable),
            "work_kind": work_kind,
            "workflow_name": "basic-workflows" if work_kind == "workflow" else "",
            "subsystem": subsystem,
            "ontology_tags": tags,
            "dependency_edges": [],
            "risk": risk,
            "parallelizable": parallelizable,
            "serialized": serialized,
            "route_hint": route_hint(subsystem, work_kind),
            "executor_recommendation": lane,
            "acceptance_test": build_acceptance_test(work_kind, title),
            "receipt_requirement": build_receipt_requirement(work_kind),
            "functionality_contract": build_functionality_contract(work_kind),
            "status": "ready",
            "detail": {
                "section_title": title,
                "source_excerpt": section[:1600],
                "source_hash": sha_text(section),
                "required_role": required_role,
            },
        }
        if idx > 1:
            edge = {"from_item_rank": idx - 1, "to_item_rank": idx, "relation": "precedes"}
            item["dependency_edges"] = [edge]
            dependency_edges.append(edge)
        items.append(item)

    batch_subsystem = "mixed" if len({item["subsystem"] for item in items}) > 1 else (items[0]["subsystem"] if items else "mixed")
    objective_summary = items[0]["detail"]["section_title"] if items else text[:160]
    batch_risk = "destructive" if any(item["risk"] == "destructive" for item in items) else ("high" if any(item["risk"] == "high" for item in items) else "medium")
    parallel_policy = "serialized" if any(item["serialized"] for item in items) and not any(item["parallelizable"] for item in items) else ("mixed" if any(item["serialized"] for item in items) and any(item["parallelizable"] for item in items) else "parallel")
    batch = {
        "batch_key": "ontobatch:" + sha_text(text)[:24],
        "source_ref": source_ref,
        "source_kind": "operator_text",
        "source_hash": sha_text(text),
        "source_excerpt": text[:MAX_SOURCE_EXCERPT],
        "objective_summary": objective_summary,
        "subsystem": batch_subsystem,
        "ontology_tags": sorted({tag for item in items for tag in item["ontology_tags"]})[:16],
        "dependency_edges": dependency_edges,
        "risk": batch_risk,
        "parallel_policy": parallel_policy,
        "planner_groups": _planner_groups(items),
        "selected_lanes": selected_lanes,
        "missing_executor_roles": missing_roles,
        "executor_recommendation": {
            "selected_model_roles": [lane.get("role") for lane in selected_lanes],
            "missing_executor_roles": missing_roles,
            "status": "blocked" if missing_roles and not selected_lanes else ("degraded" if missing_roles else "ready"),
        },
        "acceptance_test": items[0]["acceptance_test"] if items else "",
        "receipt_requirement": "receipt-gated current todo batch",
        "functionality_contract": "Preserve live API/manual truth while adding fast work decomposition and batch planning.",
        "workflow_count": sum(1 for item in items if item["work_kind"] == "workflow"),
        "workflows_preserved": any(item["workflow_name"] == "basic-workflows" for item in items),
        "batch_kind": "workflow_batch" if any(item["work_kind"] == "workflow" for item in items) else "ontology_batch",
        "status": "ready",
        "parallelizable_count": sum(1 for item in items if item["parallelizable"]),
        "serialized_count": sum(1 for item in items if item["serialized"]),
        "detail": {
            "source_ref": source_ref,
            "section_count": len(items),
            "source_char_count": len(text),
            "compiler": "scripts/ontology_work_compiler.py",
        },
    }
    return {
        "schema": "lucidota.ontology_work_compiler.v1",
        "generated_at": now_z(),
        "source_ref": source_ref,
        "batch": batch,
        "items": items,
        "selected_lanes": selected_lanes,
        "missing_executor_roles": missing_roles,
    }


def _planner_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for item in items:
        groups.setdefault(item["planner_group"], []).append(int(item["item_rank"]))
    return [
        {
            "planner_group": group,
            "item_ranks": ranks,
            "mode": "serialized" if group == "serialized_core" else "parallel",
            "reason": "DB migrations/shared core serialize; audits/tests/classification parallelize",
        }
        for group, ranks in sorted(groups.items())
    ]


def persist_batch(batch_payload: dict[str, Any], *, base_url: str = indy_runtime_broker.DEFAULT_BASE_URL, db_url: str = DB_URL) -> dict[str, Any]:
    try:
        import psycopg  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"psycopg_missing:{exc}") from exc

    batch = batch_payload["batch"]
    items = batch_payload["items"]
    with psycopg.connect(db_url, connect_timeout=5) as conn, conn.cursor() as cur:
        execute_with_bind_guard(
            cur,
            """
            INSERT INTO lucidota_control.ontology_work_batch
              (batch_key, source_ref, source_kind, source_hash, source_excerpt, objective_summary, subsystem,
               ontology_tags, dependency_edges, risk, parallel_policy, planner_groups, selected_lanes,
               missing_executor_roles, executor_recommendation, acceptance_test, receipt_requirement,
               functionality_contract, workflow_count, workflows_preserved, batch_kind, status, detail)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s, %s::text[], %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, %s::text[],
               %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (batch_key) DO UPDATE SET
              source_ref = EXCLUDED.source_ref,
              source_kind = EXCLUDED.source_kind,
              source_hash = EXCLUDED.source_hash,
              source_excerpt = EXCLUDED.source_excerpt,
              objective_summary = EXCLUDED.objective_summary,
              subsystem = EXCLUDED.subsystem,
              ontology_tags = EXCLUDED.ontology_tags,
              dependency_edges = EXCLUDED.dependency_edges,
              risk = EXCLUDED.risk,
              parallel_policy = EXCLUDED.parallel_policy,
              planner_groups = EXCLUDED.planner_groups,
              selected_lanes = EXCLUDED.selected_lanes,
              missing_executor_roles = EXCLUDED.missing_executor_roles,
              executor_recommendation = EXCLUDED.executor_recommendation,
              acceptance_test = EXCLUDED.acceptance_test,
              receipt_requirement = EXCLUDED.receipt_requirement,
              functionality_contract = EXCLUDED.functionality_contract,
              workflow_count = EXCLUDED.workflow_count,
              workflows_preserved = EXCLUDED.workflows_preserved,
              batch_kind = EXCLUDED.batch_kind,
              status = EXCLUDED.status,
              detail = EXCLUDED.detail,
              updated_at = now()
            RETURNING batch_uuid::text
            """,
            (
                batch["batch_key"],
                batch["source_ref"],
                batch["source_kind"],
                batch["source_hash"],
                batch["source_excerpt"],
                batch["objective_summary"],
                batch["subsystem"],
                batch["ontology_tags"],
                json.dumps(batch["dependency_edges"], sort_keys=True),
                batch["risk"],
                batch["parallel_policy"],
                json.dumps(batch["planner_groups"], sort_keys=True),
                json.dumps(batch["selected_lanes"], sort_keys=True),
                batch["missing_executor_roles"],
                json.dumps(batch["executor_recommendation"], sort_keys=True),
                batch["acceptance_test"],
                batch["receipt_requirement"],
                batch["functionality_contract"],
                batch["workflow_count"],
                batch["workflows_preserved"],
                batch["batch_kind"],
                batch["status"],
                json.dumps(batch["detail"], sort_keys=True),
            ),
        )
        batch_uuid = cur.fetchone()[0]
        execute_with_bind_guard(cur, "DELETE FROM lucidota_control.ontology_work_item WHERE batch_uuid = %s::uuid", (batch_uuid,))
        for item in items:
            execute_with_bind_guard(
                cur,
                """
                INSERT INTO lucidota_control.ontology_work_item
                  (batch_uuid, item_rank, planner_group, work_kind, workflow_name, subsystem, ontology_tags, dependency_edges,
                   risk, parallelizable, serialized, route_hint, executor_recommendation, acceptance_test,
                   receipt_requirement, functionality_contract, status, detail)
                VALUES
                  (%s::uuid, %s, %s, %s, %s, %s, %s::text[], %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (batch_uuid, item_rank) DO UPDATE SET
                  planner_group = EXCLUDED.planner_group,
                  work_kind = EXCLUDED.work_kind,
                  workflow_name = EXCLUDED.workflow_name,
                  subsystem = EXCLUDED.subsystem,
                  ontology_tags = EXCLUDED.ontology_tags,
                  dependency_edges = EXCLUDED.dependency_edges,
                  risk = EXCLUDED.risk,
                  parallelizable = EXCLUDED.parallelizable,
                  serialized = EXCLUDED.serialized,
                  route_hint = EXCLUDED.route_hint,
                  executor_recommendation = EXCLUDED.executor_recommendation,
                  acceptance_test = EXCLUDED.acceptance_test,
                  receipt_requirement = EXCLUDED.receipt_requirement,
                  functionality_contract = EXCLUDED.functionality_contract,
                  status = EXCLUDED.status,
                  detail = EXCLUDED.detail,
                  updated_at = now()
                """,
                (
                    batch_uuid,
                    item["item_rank"],
                    item["planner_group"],
                    item["work_kind"],
                    item["workflow_name"],
                    item["subsystem"],
                    item["ontology_tags"],
                    json.dumps(item["dependency_edges"], sort_keys=True),
                    item["risk"],
                    item["parallelizable"],
                    item["serialized"],
                    item["route_hint"],
                    json.dumps(item["executor_recommendation"], sort_keys=True),
                    item["acceptance_test"],
                    item["receipt_requirement"],
                    item["functionality_contract"],
                    item["status"],
                    json.dumps(item["detail"], sort_keys=True),
                ),
            )
        conn.commit()

    return {
        "schema": batch_payload["schema"],
        "generated_at": batch_payload["generated_at"],
        "batch": {**batch, "batch_uuid": batch_uuid},
        "items": items,
        "selected_lanes": batch_payload["selected_lanes"],
        "missing_executor_roles": batch_payload["missing_executor_roles"],
    }


def compile_work_batch(operator_text: str, *, base_url: str = indy_runtime_broker.DEFAULT_BASE_URL, source_ref: str = "operator_turn") -> dict[str, Any]:
    return compile_sections(operator_text, base_url=base_url, source_ref=source_ref)


def compile_and_persist(operator_text: str, *, base_url: str = indy_runtime_broker.DEFAULT_BASE_URL, source_ref: str = "operator_turn") -> dict[str, Any]:
    payload = compile_work_batch(operator_text, base_url=base_url, source_ref=source_ref)
    return persist_batch(payload, base_url=base_url)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile messy operator text into ontology work batches.")
    ap.add_argument("--text", default="")
    ap.add_argument("--input-file")
    ap.add_argument("--source-ref", default="operator_turn")
    ap.add_argument("--base-url", default=indy_runtime_broker.DEFAULT_BASE_URL)
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
    else:
        text = args.text or sys.stdin.read()
    payload = compile_work_batch(text, base_url=args.base_url, source_ref=args.source_ref)
    if args.persist:
        payload = compile_and_persist(text, base_url=args.base_url, source_ref=args.source_ref)

    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
