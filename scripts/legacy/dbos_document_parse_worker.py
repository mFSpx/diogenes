#!/usr/bin/env python3
"""DBOS wrapper for document parse ingestion.

Queues document parser jobs, consumes one job safely, calls the production
local parser ingestion path, and records DBOS queue events. Parser output is
NOT truth and canonical graph writes are never performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "dbos"
SCHEMAS = [
    ROOT / "06_SCHEMA" / "035_dbos_queue_spine.sql",
    ROOT / "06_SCHEMA" / "039_dbos_real_work_loop.sql",
    ROOT / "06_SCHEMA" / "045_document_ingestion_pipeline.sql",
    ROOT / "06_SCHEMA" / "076_dbos_document_parse_worker.sql",
]
QUEUE = "document_parse"
WORKFLOW = "dbos-document-parse-ingest"
JOB_KIND = "document_parse_ingest"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_db(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> str:
    return (payload or {}).get("storage_database_url") or args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"dbos_document_parse_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    report = {"action": "init_schema", "execute_performed": bool(args.execute), "schemas": [rel(s) for s in SCHEMAS]}
    if args.execute:
        with psycopg.connect(state_db(args)) as state:
            with state.cursor() as cur:
                for schema in [SCHEMAS[0], SCHEMAS[1], SCHEMAS[3]]:
                    cur.execute(schema.read_text(encoding="utf-8"))
            state.commit()
        with psycopg.connect(storage_db(args)) as storage:
            with storage.cursor() as cur:
                cur.execute(SCHEMAS[2].read_text(encoding="utf-8"))
            storage.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", report)
    return 0


def enqueue(args: argparse.Namespace) -> int:
    path = Path(args.input).expanduser().resolve()
    blockers: list[str] = []
    source_sha = ""
    if not path.exists() or not path.is_file():
        blockers.append("input_file_missing")
    else:
        source_sha = sha_file(path)
    payload = {
        "input_path": rel(path),
        "source_sha256": source_sha,
        "parser_backend": args.parser_backend,
        "storage_database_url": storage_db(args),
        "truth_status": "not_truth_evidence_candidate",
        "canonical_graph_write_allowed": False,
    }
    idem = args.idempotency_key or f"document_parse:{source_sha}:{args.parser_backend}"
    report = {"action": "enqueue", "execute_performed": bool(args.execute), "input_path": rel(path), "source_sha256": source_sha, "idempotency_key": idem, "inserted_new": False, "blockers": blockers}
    if blockers or not args.execute:
        write_report("enqueue_dry_run" if not args.execute else "enqueue_blocked", report)
        return 0 if not blockers else 2
    with psycopg.connect(state_db(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lucidota.actor_role='foreman'")
            cur.execute(
                """
                INSERT INTO lucidota_control.dbos_queue_job(queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at
                RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                """,
                (QUEUE, WORKFLOW, JOB_KIND, idem, json.dumps(payload), args.priority, args.max_attempts, json.dumps({"source": "scripts/dbos_document_parse_worker.py"})),
            )
            job_uuid, inserted_new = cur.fetchone()
            if inserted_new:
                cur.execute(
                    "INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s,%s,'enqueued','dbos_document_parse_worker',%s::jsonb)",
                    (job_uuid, QUEUE, json.dumps({"source_sha256": source_sha, "input_path": rel(path)})),
                )
        conn.commit()
    report.update({"job_uuid": job_uuid, "inserted_new": bool(inserted_new)})
    write_report("enqueue_execute", report)
    print(f"JOB_UUID={job_uuid}")
    print(f"INSERTED_NEW={str(bool(inserted_new)).lower()}")
    return 0


def extract_run_uuid(stdout: str) -> str | None:
    m = re.search(r"^RUN_UUID=([0-9a-fA-F-]+)$", stdout, flags=re.M)
    return m.group(1) if m else None


def worker_once(args: argparse.Namespace) -> int:
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    report: dict[str, Any] = {"action": "worker_once", "execute_performed": bool(args.execute), "worker_id": worker_id, "job_processed": False, "graph_writes_performed": False, "blockers": []}
    with psycopg.connect(state_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if not args.execute:
                cur.execute("SELECT job_uuid::text,payload,status::text FROM lucidota_control.dbos_queue_job WHERE queue_name=%s AND job_kind=%s AND status='queued' ORDER BY priority ASC, created_at ASC LIMIT 1", (QUEUE, JOB_KIND))
                row = cur.fetchone()
                report["would_process"] = dict(row) if row else None
                write_report("worker_dry_run", report)
                return 0
            cur.execute("SET LOCAL lucidota.actor_role='worker'")
            cur.execute(
                """
                SELECT job_uuid::text, payload, idempotency_key, attempt_count, max_attempts
                FROM lucidota_control.dbos_queue_job
                WHERE queue_name=%s AND job_kind=%s AND status='queued' AND run_after <= now()
                ORDER BY priority ASC, created_at ASC
                FOR UPDATE SKIP LOCKED LIMIT 1
                """,
                (QUEUE, JOB_KIND),
            )
            row = cur.fetchone()
            if not row:
                report["no_job_available"] = True
                write_report("worker_execute", report)
                return 0
            job_uuid = row["job_uuid"]
            payload = row["payload"]
            cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='running', locked_by=%s, locked_at=now(), leased_by=%s, lease_expires_at=now()+interval '5 minutes', attempt_count=attempt_count+1 WHERE job_uuid=%s", (worker_id, worker_id, job_uuid))
            cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s,%s,'started','dbos_document_parse_worker',%s::jsonb)", (job_uuid, QUEUE, json.dumps({"worker_id": worker_id})))
            conn.commit()
    # Run parser outside the queue row lock.
    input_path = Path(payload["input_path"])
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    cmd = [sys.executable, "scripts/document_parse_ingest.py", "--database-url", storage_db(args, payload), "ingest", "--execute", "--input", str(input_path)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout_seconds)
    run_uuid = extract_run_uuid(proc.stdout)
    ok = proc.returncode == 0 and bool(run_uuid)
    result = {"command": " ".join(cmd), "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:], "run_uuid": run_uuid, "truth_status": "not_truth_evidence_candidate"}
    status = "succeeded" if ok else ("dead_lettered" if int(row["attempt_count"]) + 1 >= int(row["max_attempts"]) else "failed")
    with psycopg.connect(state_db(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lucidota.actor_role='worker'")
            cur.execute("UPDATE lucidota_control.dbos_queue_job SET status=%s, result=%s::jsonb, completed_at=CASE WHEN %s THEN now() ELSE completed_at END, last_error=%s WHERE job_uuid=%s", (status, json.dumps(result), ok, "" if ok else (proc.stderr[-1000:] or "document_parse_failed"), job_uuid))
            cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid,queue_name,event_kind,event_source,detail) VALUES (%s,%s,%s,'dbos_document_parse_worker',%s::jsonb)", (job_uuid, QUEUE, "succeeded" if ok else status, json.dumps({"run_uuid": run_uuid, "returncode": proc.returncode})))
            if not ok and status == "dead_lettered":
                cur.execute(
                    """
                    INSERT INTO lucidota_control.dbos_queue_dead_letter(job_uuid,queue_name,workflow_name,job_kind,idempotency_key,error_kind,error_message,attempt_count,payload_sha256,context)
                    VALUES (%s,%s,%s,%s,%s,'document_parse_error',%s,%s,%s,%s::jsonb)
                    ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET last_seen_at=now(), attempt_count=EXCLUDED.attempt_count, error_message=EXCLUDED.error_message
                    """,
                    (job_uuid, QUEUE, WORKFLOW, JOB_KIND, row["idempotency_key"], proc.stderr[-1000:] or "document_parse_failed", int(row["attempt_count"]) + 1, sha_obj(payload), json.dumps({"input_path": payload.get("input_path")})),
                )
        conn.commit()
    report.update({"job_processed": True, "job_uuid": job_uuid, "status": status, "run_uuid": run_uuid, "parser_returncode": proc.returncode, "parser_stdout_tail": proc.stdout[-1000:], "parser_stderr_tail": proc.stderr[-1000:]})
    write_report("worker_execute", report)
    print(f"JOB_UUID={job_uuid}")
    print(f"STATUS={status}")
    if run_uuid:
        print(f"RUN_UUID={run_uuid}")
    return 0 if ok else 3


def audit(args: argparse.Namespace) -> int:
    with psycopg.connect(state_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status::text, count(*) AS n FROM lucidota_control.dbos_queue_job WHERE queue_name=%s GROUP BY status::text ORDER BY status::text", (QUEUE,))
            queue_counts = {r["status"]: int(r["n"]) for r in cur.fetchall()}
    with psycopg.connect(storage_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM lucidota_korpus.document_parse_run")
            run_count = int(cur.fetchone()["n"])
            cur.execute("SELECT count(*) AS n FROM lucidota_korpus.document_parse_span")
            span_count = int(cur.fetchone()["n"])
    write_report("audit", {"action": "audit", "queue_counts": queue_counts, "document_parse_run_count": run_count, "document_parse_span_count": span_count, "graph_writes_performed": False})
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="DBOS document parse worker")
    p.add_argument("--state-database-url")
    p.add_argument("--storage-database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("enqueue"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--input", required=True); sp.add_argument("--parser-backend", default="local_text_v1"); sp.add_argument("--priority", type=int, default=50); sp.add_argument("--max-attempts", type=int, default=3); sp.add_argument("--idempotency-key"); sp.set_defaults(func=enqueue)
    sp = sub.add_parser("worker-once"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--worker-id"); sp.add_argument("--timeout-seconds", type=int, default=120); sp.set_defaults(func=worker_once)
    sp = sub.add_parser("audit"); sp.set_defaults(func=audit)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
