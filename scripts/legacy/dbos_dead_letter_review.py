#!/usr/bin/env python3
"""DBOS queue dead-letter review CLI.

Lists, classifies, and retries DBOS queue dead-letter rows only with explicit
--execute for mutations. It never deletes DLQ rows and never marks a DLQ row
resolved as a side effect of retry creation.
"""
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
SCHEMA = ROOT / "06_SCHEMA/084_dbos_dead_letter_review.sql"
OUT = ROOT / "05_OUTPUTS/dbos"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_report(action: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"dbos_dead_letter_review_{action}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"action": "init_schema", "execute_performed": bool(args.execute), "schema": rel(SCHEMA)})
    print("DLQ_REVIEW_SCHEMA=" + ("APPLIED" if args.execute else "DRY_RUN"))
    return 0


def list_rows(args: argparse.Namespace) -> int:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT dlq.dead_letter_uuid::text, dlq.job_uuid::text, dlq.queue_name, dlq.workflow_name,
                       dlq.job_kind, dlq.idempotency_key, dlq.error_kind, dlq.error_message,
                       dlq.attempt_count, dlq.resolved, dlq.first_seen_at, dlq.last_seen_at,
                       coalesce(jsonb_agg(jsonb_build_object('action',r.action,'classification',r.classification,'created_at',r.created_at) ORDER BY r.created_at DESC) FILTER (WHERE r.review_uuid IS NOT NULL), '[]'::jsonb) AS reviews
                FROM lucidota_control.dbos_queue_dead_letter dlq
                LEFT JOIN lucidota_control.dbos_dead_letter_review r ON r.dead_letter_uuid=dlq.dead_letter_uuid
                WHERE (%s OR dlq.resolved=false)
                GROUP BY dlq.dead_letter_uuid
                ORDER BY dlq.last_seen_at DESC
                LIMIT %s
                """,
                (args.include_resolved, args.limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
    report = {"action": "list", "execute_performed": False, "rows": rows, "unresolved_count": sum(1 for r in rows if not r["resolved"]), "db_writes_performed": False, "graph_writes_performed": False}
    write_report("list", report)
    print(f"DLQ_ROWS={len(rows)}")
    return 0


def classify(args: argparse.Namespace) -> int:
    detail = {"source": "scripts/dbos_dead_letter_review.py", "dead_letter_uuid": args.dead_letter_uuid, "classification": args.classification}
    review_uuid = None
    if args.execute:
        with psycopg.connect(db(args), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.dbos_dead_letter_review(dead_letter_uuid, action, classification, execute_performed, note, detail)
                    VALUES (%s, 'classify', %s, true, %s, %s::jsonb)
                    RETURNING review_uuid::text
                    """,
                    (args.dead_letter_uuid, args.classification, args.note or "", json.dumps(detail)),
                )
                review_uuid = cur.fetchone()["review_uuid"]
            conn.commit()
    report = {"action": "classify", "execute_performed": bool(args.execute), "dead_letter_uuid": args.dead_letter_uuid, "classification": args.classification, "review_uuid": review_uuid, "dlq_row_mutated": False, "db_writes_performed": bool(args.execute), "graph_writes_performed": False}
    write_report("classify_execute" if args.execute else "classify_dry_run", report)
    print("DLQ_CLASSIFY=" + ("EXECUTED" if args.execute else "DRY_RUN"))
    return 0


def retry(args: argparse.Namespace) -> int:
    retry_job = None
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT dlq.dead_letter_uuid::text, dlq.queue_name, dlq.workflow_name, dlq.job_kind,
                       dlq.idempotency_key, j.payload, j.priority, j.max_attempts
                FROM lucidota_control.dbos_queue_dead_letter dlq
                JOIN lucidota_control.dbos_queue_job j ON j.job_uuid=dlq.job_uuid
                WHERE dlq.dead_letter_uuid=%s AND dlq.resolved=false
                """,
                (args.dead_letter_uuid,),
            )
            source = cur.fetchone()
            if not source:
                report = {"action": "retry", "execute_performed": bool(args.execute), "dead_letter_uuid": args.dead_letter_uuid, "blockers": ["unresolved_dead_letter_not_found"]}
                write_report("retry_fail", report)
                print("DLQ_RETRY=FAIL")
                return 3
            retry_key = f"retry:{source['dead_letter_uuid']}:{sha_obj(source['payload'])[:12]}"
            if args.execute:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.dbos_queue_job(queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                    RETURNING job_uuid::text, status::text
                    """,
                    (source["queue_name"], source["workflow_name"], source["job_kind"], retry_key, json.dumps(source["payload"]), int(source["priority"]), int(source["max_attempts"]), json.dumps({"retry_of_dead_letter_uuid": source["dead_letter_uuid"]})),
                )
                retry_job = dict(cur.fetchone())
                cur.execute(
                    """
                    INSERT INTO lucidota_control.dbos_dead_letter_review(dead_letter_uuid, action, classification, execute_performed, retry_job_uuid, note, detail)
                    VALUES (%s, 'retry_enqueued', %s, true, %s, %s, %s::jsonb)
                    """,
                    (args.dead_letter_uuid, args.classification, retry_job["job_uuid"], args.note or "", json.dumps({"retry_idempotency_key": retry_key})),
                )
                conn.commit()
    report = {"action": "retry", "execute_performed": bool(args.execute), "dead_letter_uuid": args.dead_letter_uuid, "retry_idempotency_key": f"retry:{args.dead_letter_uuid}:<payload_sha_prefix>", "retry_job": retry_job, "dlq_row_mutated": False, "db_writes_performed": bool(args.execute), "graph_writes_performed": False}
    write_report("retry_execute" if args.execute else "retry_dry_run", report)
    print("DLQ_RETRY=" + ("EXECUTED" if args.execute else "DRY_RUN"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Review DBOS dead-letter rows safely")
    ap.add_argument("--database-url")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init-schema"); s.add_argument("--execute", action="store_true")
    l = sub.add_parser("list"); l.add_argument("--limit", type=int, default=25); l.add_argument("--include-resolved", action="store_true")
    c = sub.add_parser("classify"); c.add_argument("--dead-letter-uuid", required=True); c.add_argument("--classification", choices=["unknown","smoke_test","transient","permanent","operator_review","bug","external_dependency"], default="unknown"); c.add_argument("--note", default=""); c.add_argument("--execute", action="store_true")
    r = sub.add_parser("retry"); r.add_argument("--dead-letter-uuid", required=True); r.add_argument("--classification", choices=["unknown","smoke_test","transient","permanent","operator_review","bug","external_dependency"], default="unknown"); r.add_argument("--note", default=""); r.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.cmd == "init-schema": return init_schema(args)
    if args.cmd == "list": return list_rows(args)
    if args.cmd == "classify": return classify(args)
    if args.cmd == "retry": return retry(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
