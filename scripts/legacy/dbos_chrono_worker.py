#!/usr/bin/env python3
"""DBOS queue-spine wrapper for Chrono-Ledger.

The wrapper observes Chrono service health and writes DBOS queue/job/workflow
receipts. It never deletes, overwrites, invalidates, or inserts temporal claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "dbos"
STATE_SCHEMA = ROOT / "06_SCHEMA" / "035_dbos_queue_spine.sql"
CHRONO_WRAPPER_SCHEMA = ROOT / "06_SCHEMA" / "036_dbos_chrono_wrapper.sql"
SERVICE_CHECK = ROOT / "scripts" / "check_chrono_ledger_service.sh"
QUEUE_NAME = "chrono"
JOB_KIND = "chrono_health_check"
WORKFLOW_NAME = "dbos-chrono-health-check"
CANONICAL_GRAPH_TABLES = ["lucidota_go.graph_item", "lucidota_go.graph_edge", "lucidota_go.graph_journal"]
TEMPORAL_TABLES = ["lucidota_korpus.temporal_claim", "lucidota_korpus.resolved_chrono_timeline"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def redact(url: str | None) -> str:
    if not url:
        return "unset"
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def state_url(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_url(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> str:
    if args.storage_database_url:
        return args.storage_database_url
    if payload and payload.get("storage_database_url"):
        return str(payload["storage_database_url"])
    return os.environ.get("CHRONO_DATABASE_URL") or "postgresql:///lucidota_storage"


def first_value(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def count_table(conn, table: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        if first_value(cur.fetchone()) is None:
            return None
        cur.execute(f"SELECT count(*) FROM {table}")
        return int(first_value(cur.fetchone()))


def count_tables(conn, tables: list[str]) -> dict[str, int | None]:
    return {table: count_table(conn, table) for table in tables}


def write_report(action: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"dbos_chrono_wrapper_{action}_{stamp()}.json"
    report["report_path"] = str(out.relative_to(ROOT))
    out.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return out


def run_service_check(db_url: str) -> dict[str, Any]:
    if not SERVICE_CHECK.exists():
        return {"script_exists": False, "returncode": 127, "stdout_tail": "", "stderr_tail": "script_missing"}
    env = os.environ.copy()
    env["DATABASE_URL"] = db_url
    proc = subprocess.run([str(SERVICE_CHECK)], cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    return {
        "script_exists": True,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def chrono_db_health(db_url: str) -> dict[str, Any]:
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH violations AS (
                  SELECT r.file_uuid
                  FROM lucidota_korpus.resolved_chrono_timeline r
                  JOIN lucidota_korpus.temporal_claim tc ON tc.file_uuid = r.file_uuid
                  WHERE tc.trust_weight > r.confidence_score
                     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp < r.resolved_timestamp)
                ) SELECT COUNT(*) AS ranking_violations FROM violations
            """)
            ranking_violations = int(cur.fetchone()["ranking_violations"])
            cur.execute("""
                SELECT cursor_name,last_file_uuid::text,last_file_first_seen_at::text,processed_count,
                       last_replay_started_at::text,last_replay_finished_at::text,updated_at::text
                FROM lucidota_korpus.chrono_replay_cursor
                ORDER BY cursor_name
            """)
            cursors = list(cur.fetchall())
            cur.execute("SELECT resolved, COUNT(*) AS count FROM lucidota_korpus.chrono_dead_letter GROUP BY resolved ORDER BY resolved")
            dead_letters = list(cur.fetchall())
            cur.execute("""
                SELECT COUNT(*) AS pending_target_count
                FROM lucidota_korpus.file_object f
                WHERE NOT EXISTS (SELECT 1 FROM lucidota_korpus.temporal_claim tc WHERE tc.file_uuid=f.file_uuid)
            """)
            pending = int(cur.fetchone()["pending_target_count"])
            cur.execute("SELECT COUNT(*) AS total_claims, COUNT(DISTINCT file_uuid) AS files_covered FROM lucidota_korpus.temporal_claim")
            claims = cur.fetchone()
            temporal_counts = count_tables(conn, TEMPORAL_TABLES)
    return {
        "ranking_violations": ranking_violations,
        "replay_cursor_status": cursors,
        "dead_letter_status": dead_letters,
        "pending_target_count": pending,
        "total_claims": int(claims["total_claims"]),
        "files_covered": int(claims["files_covered"]),
        "temporal_counts": temporal_counts,
    }


def chrono_health(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    db_url = storage_url(args, payload)
    service = run_service_check(db_url)
    db_health = chrono_db_health(db_url)
    active = service["returncode"] == 0
    if not active:
        blockers.append("chrono_service_check_failed")
    if db_health["ranking_violations"] != 0:
        blockers.append("ranking_violations_nonzero")
    return {
        "storage_database_url": redact(db_url),
        "service_check": service,
        "service_still_running": active,
        "db_health": db_health,
        "temporal_claims_mutated_by_wrapper": False,
    }, blockers


def init_schema(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    url = state_url(args)
    result: dict[str, Any] = {"state_database_url": redact(url), "execute_performed": False}
    if not execute:
        result["would_apply"] = [str(STATE_SCHEMA.relative_to(ROOT)), str(CHRONO_WRAPPER_SCHEMA.relative_to(ROOT))]
        return result, blockers
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(STATE_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(CHRONO_WRAPPER_SCHEMA.read_text(encoding="utf-8"))
            cur.execute("SELECT count(*) FROM lucidota_control.dbos_queue WHERE queue_name='chrono'")
            result["chrono_queue_registered"] = int(cur.fetchone()[0]) == 1
            cur.execute("SELECT count(*) FROM lucidota_control.workflow_registry WHERE workflow_name=%s", (WORKFLOW_NAME,))
            result["workflow_registered"] = int(cur.fetchone()[0]) == 1
        conn.commit()
    result["execute_performed"] = True
    return result, blockers


def enqueue(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    url = state_url(args)
    payload = {
        "check_kind": "chrono_health_check",
        "storage_database_url": storage_url(args),
        "service_unit": "lucidota-chrono-ledger.service",
        "temporal_conservation_law": "observe_only_no_temporal_claim_mutation",
    }
    idempotency_key = args.idempotency_key or sha256_obj({"queue": QUEUE_NAME, "job_kind": JOB_KIND, "payload": {k: v for k, v in payload.items() if k != "storage_database_url"}})
    result: dict[str, Any] = {"state_database_url": redact(url), "queue": QUEUE_NAME, "workflow": WORKFLOW_NAME, "job_kind": JOB_KIND, "idempotency_key": idempotency_key, "payload_sha256": sha256_obj(payload), "execute_performed": False, "job_uuid": None, "inserted_new": False}
    if not execute:
        return result, blockers
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            graph_before = {}
            try:
                graph_before = count_tables(cur.connection, CANONICAL_GRAPH_TABLES)
            except Exception as exc:
                graph_before = {"error": str(exc)}
            cur.execute("""
                INSERT INTO lucidota_control.dbos_queue_job
                  (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                RETURNING job_uuid, (xmax = 0) AS inserted_new
            """, (QUEUE_NAME, WORKFLOW_NAME, JOB_KIND, idempotency_key, json.dumps(payload), args.priority, args.max_attempts, json.dumps({"source":"dbos_chrono_worker"})))
            job_uuid, inserted_new = cur.fetchone()
            if inserted_new:
                cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,%s,'enqueued','dbos_chrono_worker',%s::jsonb)", (job_uuid, QUEUE_NAME, json.dumps({"job_kind":JOB_KIND, "idempotency_key":idempotency_key})))
            result.update({"execute_performed": True, "job_uuid": str(job_uuid), "inserted_new": bool(inserted_new), "canonical_graph_counts_before": graph_before})
        conn.commit()
    return result, blockers


def worker_once(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    url = state_url(args)
    worker_id = args.worker_id or f"dbos_chrono:{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {"state_database_url": redact(url), "queue": QUEUE_NAME, "worker_id": worker_id, "execute_performed": False, "job_processed": False}
    with psycopg.connect(url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, status::text, payload
                FROM lucidota_control.dbos_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after <= now()
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            """, (QUEUE_NAME,))
            preview = cur.fetchone()
            if not execute:
                result["would_process"] = dict(preview) if preview else None
                return result, blockers
            if not preview:
                result["no_job_available"] = True
                return result, blockers
            job_uuid = preview["job_uuid"]
            payload = preview["payload"]
            if preview["job_kind"] != JOB_KIND:
                blockers.append("unexpected_job_kind")
                return result, blockers
            graph_before: dict[str, Any] = {}
            try:
                graph_before = count_tables(conn, CANONICAL_GRAPH_TABLES)
            except Exception as exc:
                graph_before = {"error": str(exc)}
            cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='running', leased_by=%s, lease_expires_at=now()+interval '5 minutes', attempt_count=attempt_count+1, updated_at=now() WHERE job_uuid=%s::uuid", (worker_id, job_uuid))
            cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s::uuid,%s,'started','dbos_chrono_worker',%s::jsonb)", (job_uuid, QUEUE_NAME, json.dumps({"worker_id": worker_id})))
            health, health_blockers = chrono_health(args, payload)
            ok = not health_blockers
            status = "succeeded" if ok else "failed"
            cur.execute("""
                UPDATE lucidota_control.dbos_queue_job
                SET status=%s, result=%s::jsonb, completed_at=CASE WHEN %s THEN now() ELSE completed_at END,
                    updated_at=now(), last_error=%s
                WHERE job_uuid=%s::uuid
            """, (status, json.dumps(health, default=str), ok, ";".join(health_blockers), job_uuid))
            cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s::uuid,%s,%s,'dbos_chrono_worker',%s::jsonb)", (job_uuid, QUEUE_NAME, status, json.dumps({"health_blockers": health_blockers, "ranking_violations": health.get("db_health",{}).get("ranking_violations")})))
            cur.execute("""
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES (%s,%s,'dbos_chrono_wrapper',%s,'dbos_chrono_worker',%s::jsonb)
                RETURNING event_id::text
            """, (WORKFLOW_NAME, job_uuid, status, json.dumps({"queue":"chrono", "job_uuid": job_uuid, "health": health}, default=str)))
            event_id = first_value(cur.fetchone())
            if not ok:
                cur.execute("""
                    INSERT INTO lucidota_control.dbos_queue_dead_letter
                      (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, error_kind, error_message, attempt_count, payload_sha256, context)
                    SELECT job_uuid, queue_name, workflow_name, job_kind, idempotency_key, 'chrono_health_failed', %s, attempt_count, %s, %s::jsonb
                    FROM lucidota_control.dbos_queue_job WHERE job_uuid=%s::uuid
                    ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET
                      error_message=EXCLUDED.error_message,last_seen_at=now(),context=EXCLUDED.context
                """, (";".join(health_blockers), sha256_obj(payload), json.dumps(health, default=str), job_uuid))
            graph_after: dict[str, Any] = {}
            try:
                graph_after = count_tables(conn, CANONICAL_GRAPH_TABLES)
            except Exception as exc:
                graph_after = {"error": str(exc)}
            result.update({"execute_performed": True, "job_processed": True, "job_uuid": job_uuid, "workflow_event_id": event_id, "status": status, "health": health, "canonical_graph_counts_before": graph_before, "canonical_graph_counts_after": graph_after, "canonical_graph_writes_performed": graph_before != graph_after, "temporal_claims_mutated_by_wrapper": False})
            if graph_before != graph_after:
                blockers.append("canonical_graph_counts_changed")
            blockers.extend(health_blockers)
        if blockers:
            conn.commit()  # keep failure evidence/dead-letter; no canonical graph mutation is permitted above.
        else:
            conn.commit()
    return result, blockers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["audit", "init-schema", "enqueue-health-check", "worker-once"], required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--state-database-url", default=os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
    ap.add_argument("--storage-database-url", default=os.environ.get("CHRONO_DATABASE_URL", "postgresql:///lucidota_storage"))
    ap.add_argument("--queue", default=QUEUE_NAME)
    ap.add_argument("--idempotency-key")
    ap.add_argument("--priority", type=int, default=50)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--worker-id")
    args = ap.parse_args()
    execute = bool(args.execute)
    blockers: list[str] = []
    try:
        if args.action == "init-schema":
            action_result, blockers = init_schema(args, execute)
        elif args.action == "enqueue-health-check":
            action_result, blockers = enqueue(args, execute)
        elif args.action == "worker-once":
            action_result, blockers = worker_once(args, execute)
        else:
            health, hblockers = chrono_health(args)
            action_result = {"health": health, "execute_performed": False}
            blockers = hblockers
    except Exception as exc:
        action_result = {}
        blockers = [f"exception:{exc}"]
    report = {
        "schema": "lucidota.dbos_chrono_wrapper.report.v1",
        "generated_at": now_iso(),
        "action": args.action,
        "mode": "execute" if execute else "dry_run",
        "execute_requested": execute,
        "state_database_url": redact(state_url(args)),
        "storage_database_url": redact(storage_url(args)),
        "action_result": action_result,
        "db_writes_performed": bool(action_result.get("execute_performed")) if isinstance(action_result, dict) else False,
        "temporal_claims_mutated_by_wrapper": False,
        "canonical_graph_writes_performed": bool(action_result.get("canonical_graph_writes_performed")) if isinstance(action_result, dict) else False,
        "blockers": blockers,
    }
    write_report(args.action, report)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
