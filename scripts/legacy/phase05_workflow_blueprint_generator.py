#!/usr/bin/env python3
"""Generate DBOS implementation-candidate work orders from Phase 0.5 workflow blueprints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "phase05"
STATE_SCHEMAS = [
    ROOT / "06_SCHEMA/035_dbos_queue_spine.sql",
    ROOT / "06_SCHEMA/039_dbos_real_work_loop.sql",
    ROOT / "06_SCHEMA/075_phase05_workflow_blueprint_queue.sql",
]
WORKFLOW = "phase05-workflow-blueprint-implementation-candidate"
QUEUE = "control"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"phase05_workflow_blueprint_generator_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                for schema in STATE_SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "schemas": [rel(s) for s in STATE_SCHEMAS],
    })
    return 0


def fetch_blueprints(args: argparse.Namespace) -> list[dict[str, Any]]:
    with psycopg.connect(storage_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT blueprint_uuid::text, workflow_key, title, purpose, maturity, dbos_target,
                       input_contract, output_contract, steps, queues, required_tables,
                       required_services, required_models, source_atom_uuids,
                       authority_class::text, canonical_confidence_bps, operator_confirmed
                FROM lucidota_archaeology.workflow_blueprint
                WHERE dbos_target = true
                  AND maturity IN ('recovered','drafted','implementable','implemented')
                ORDER BY operator_confirmed DESC, canonical_confidence_bps DESC, updated_at DESC
                LIMIT %s
                """,
                (args.limit,),
            )
            return [dict(row) for row in cur.fetchall()]


def queue_candidates(args: argparse.Namespace) -> int:
    blueprints = fetch_blueprints(args)
    queued: list[dict[str, Any]] = []
    if args.execute and blueprints:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                for bp in blueprints:
                    idem = f"phase05_workflow_blueprint:{bp['blueprint_uuid']}:v1"
                    payload = {
                        "handler": "noop",
                        "message": f"Phase 0.5 workflow blueprint candidate: {bp['workflow_key']}",
                        "blueprint": bp,
                        "truth_status": "workflow_candidate_not_implemented",
                        "graph_promotion_status": "not_promoted",
                        "requires_operator_priority": True,
                    }
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.dbos_queue_job(
                          queue_name, workflow_name, job_kind, idempotency_key,
                          payload, priority, max_attempts, detail
                        )
                        VALUES (%s,%s,'phase05_workflow_blueprint_candidate',%s,%s::jsonb,%s,2,%s::jsonb)
                        ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET
                          updated_at=lucidota_control.dbos_queue_job.updated_at
                        RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                        """,
                        (
                            QUEUE,
                            WORKFLOW,
                            idem,
                            json.dumps(payload, default=str),
                            args.priority,
                            json.dumps({"script": "scripts/phase05_workflow_blueprint_generator.py", "payload_sha256": sha_obj(payload)}),
                        ),
                    )
                    job_row = cur.fetchone()
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.phase05_workflow_blueprint_queue_receipt(
                          blueprint_uuid, workflow_key, job_uuid, queue_name, idempotency_key, detail
                        )
                        VALUES (%s::uuid,%s,%s::uuid,%s,%s,%s::jsonb)
                        ON CONFLICT (blueprint_uuid) DO UPDATE SET
                          detail=lucidota_control.phase05_workflow_blueprint_queue_receipt.detail || EXCLUDED.detail
                        RETURNING receipt_uuid::text
                        """,
                        (
                            bp["blueprint_uuid"],
                            bp["workflow_key"],
                            job_row["job_uuid"],
                            QUEUE,
                            idem,
                            json.dumps({"job_inserted_new": job_row["inserted_new"]}),
                        ),
                    )
                    receipt_uuid = cur.fetchone()["receipt_uuid"]
                    if job_row["inserted_new"]:
                        cur.execute(
                            """
                            INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail)
                            VALUES (%s::uuid,%s,'enqueued','phase05_workflow_blueprint_generator',%s::jsonb)
                            """,
                            (job_row["job_uuid"], QUEUE, json.dumps({"blueprint_uuid": bp["blueprint_uuid"], "workflow_key": bp["workflow_key"]})),
                        )
                    queued.append({"blueprint_uuid": bp["blueprint_uuid"], "workflow_key": bp["workflow_key"], "job_uuid": job_row["job_uuid"], "receipt_uuid": receipt_uuid, "inserted_new": bool(job_row["inserted_new"])})
            conn.commit()
    report = {
        "action": "queue_candidates",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "blueprints_seen": len(blueprints),
        "jobs_queued": len(queued),
        "inserted_new": sum(1 for q in queued if q.get("inserted_new")),
        "queued": queued,
        "sample_blueprints": blueprints[:8],
        "blockers": [] if blueprints else ["no_workflow_blueprints"],
    }
    write_report("queue_execute" if args.execute else "queue_dry_run", report)
    print(f"BLUEPRINTS_SEEN={len(blueprints)}")
    print(f"JOBS_QUEUED={len(queued) if args.execute else 0}")
    return 0 if blueprints else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue Phase 0.5 workflow blueprints as DBOS implementation candidates")
    parser.add_argument("--state-database-url")
    parser.add_argument("--storage-database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("queue-candidates")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--priority", type=int, default=80)
    p.set_defaults(func=queue_candidates)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
