#!/usr/bin/env python3
"""LUCIDOTA ABSURD-compatible durable queue spine.

Dry-run is default for actions that can write. Execute mode writes only queue-spine
state (jobs/events/dead-letter/workflow_event), never canonical graph tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
from absurd_worker_contracts import gate_worker_payload_hygiene
SCHEMA = ROOT / "06_SCHEMA" / "035_absurd_queue_spine.sql"
REAL_WORK_LOOP_SCHEMA = ROOT / "06_SCHEMA" / "039_absurd_real_work_loop.sql"
OUT_DIR = ROOT / "05_OUTPUTS" / "absurd"
PY = Path(sys.executable)
QUEUE_TABLES = [
    "lucidota_control.absurd_queue",
    "lucidota_control.absurd_queue_job",
    "lucidota_control.absurd_queue_event",
    "lucidota_control.absurd_queue_dead_letter",
]
CANONICAL_GRAPH_TABLES = [
    "lucidota_go.graph_item",
    "lucidota_go.graph_edge",
    "lucidota_go.graph_journal",
]
ALLOWED_JOB_KINDS = {"status_ledger_check", "noop", "external_command", "momentary_flow", "bitloops_context_ingest"}
ALLOWED_EXTERNAL_COMMANDS = {
    "scripts/goal_agent_packet.py",
    "scripts/goal_dev_control.py",
    "scripts/goal_model_fabric_control.py",
    "scripts/lucidota_model_turbine_overseer.py",
    "scripts/groq_goal_delegate.py",
    "scripts/goal_model_fabric_orchestrate.py",
    "scripts/model_runner_cli.py",
    "scripts/language_router.py",
    "scripts/lucidota_usecase_proof.py",
}
MAX_PAYLOAD_JSON_BYTES = 65536


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def db_url(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def redacted(url: str) -> str:
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def connect(url: str):
    return psycopg.connect(url)


def table_count(cur, table: str) -> int | None:
    cur.execute("SELECT to_regclass(%s)", (table,))
    if cur.fetchone()[0] is None:
        return None
    cur.execute(f"SELECT count(*) FROM {table}")
    return int(cur.fetchone()[0])


def counts(cur, tables: list[str]) -> dict[str, int | None]:
    return {t: table_count(cur, t) for t in tables}


def write_report(action: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"absurd_queue_spine_{action}_{stamp()}.json"
    report["report_path"] = str(out.relative_to(ROOT))
    out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return out


def load_payload_json(raw: str) -> dict[str, Any]:
    raw_bytes = raw.encode("utf-8")
    if len(raw_bytes) > MAX_PAYLOAD_JSON_BYTES:
        raise ValueError(f"payload_json_too_large:{len(raw_bytes)}>{MAX_PAYLOAD_JSON_BYTES}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("payload_json_must_be_object")
    return payload


def apply_schema(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    url = db_url(args)
    schemas = [SCHEMA, REAL_WORK_LOOP_SCHEMA]
    result: dict[str, Any] = {"database_url": redacted(url), "schema_paths": [str(schema.relative_to(ROOT)) for schema in schemas], "execute_performed": False}
    if any(not schema.exists() for schema in schemas):
        return result, ["schema_missing"]
    if not execute:
        result["dry_run_sql_bytes"] = sum(schema.stat().st_size for schema in schemas)
        return result, blockers
    with connect(url) as conn:
        with conn.cursor() as cur:
            for schema in schemas:
                cur.execute(schema.read_text(encoding="utf-8"))
            result["queue_counts_after"] = counts(cur, QUEUE_TABLES)
        conn.commit()
    result["execute_performed"] = True
    return result, blockers


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    url = db_url(args)
    blockers: list[str] = []
    result: dict[str, Any] = {"database_url": redacted(url)}
    with connect(url) as conn:
        with conn.cursor() as cur:
            result["queue_counts"] = counts(cur, QUEUE_TABLES)
            missing = [t for t, c in result["queue_counts"].items() if c is None]
            if missing:
                blockers.append("queue_tables_missing:" + ",".join(missing))
            try:
                result["canonical_graph_counts"] = counts(cur, CANONICAL_GRAPH_TABLES)
            except Exception as exc:
                result["canonical_graph_counts_error"] = str(exc)
    return result, blockers


def enqueue(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url = db_url(args)
    blockers: list[str] = []
    try:
        payload = load_payload_json(args.payload_json)
    except Exception as exc:
        result = {
            "database_url": redacted(url),
            "queue": args.queue,
            "workflow": args.workflow,
            "job_kind": args.job_kind,
            "idempotency_key": args.idempotency_key,
            "payload_sha256": None,
            "payload_error": str(exc),
            "execute_performed": False,
            "job_uuid": None,
            "inserted_new": False,
        }
        blocker = str(exc).split(":", 1)[0] if str(exc).startswith("payload_json_too_large") else str(exc)
        if blocker not in {"payload_json_too_large", "payload_json_must_be_object"}:
            blocker = "payload_json_invalid"
        return result, [blocker]
    if args.job_kind not in ALLOWED_JOB_KINDS:
        blockers.append("job_kind_not_allowed")
    idempotency_key = args.idempotency_key or sha256_obj({"queue": args.queue, "workflow": args.workflow, "job_kind": args.job_kind, "payload": payload})
    result: dict[str, Any] = {
        "database_url": redacted(url),
        "queue": args.queue,
        "workflow": args.workflow,
        "job_kind": args.job_kind,
        "idempotency_key": idempotency_key,
        "payload_sha256": sha256_obj(payload),
        "execute_performed": False,
        "job_uuid": None,
        "inserted_new": False,
    }
    if blockers or not execute:
        return result, blockers
    with connect(url) as conn:
        with conn.cursor() as cur:
            before_graph = counts(cur, CANONICAL_GRAPH_TABLES)
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_job
                  (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                RETURNING job_uuid, (xmax = 0) AS inserted_new
                """,
                (args.queue, args.workflow, args.job_kind, idempotency_key, json.dumps(payload), args.priority, args.max_attempts, json.dumps({"source":"absurd_queue_spine"})),
            )
            job_uuid, inserted_new = cur.fetchone()
            if inserted_new:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail)
                    VALUES (%s,%s,'enqueued',%s::jsonb)
                    """,
                    (job_uuid, args.queue, json.dumps({"workflow": args.workflow, "job_kind": args.job_kind, "idempotency_key": idempotency_key})),
                )
            after_graph = counts(cur, CANONICAL_GRAPH_TABLES)
            result.update({"execute_performed": True, "job_uuid": str(job_uuid), "inserted_new": bool(inserted_new), "canonical_graph_counts_before": before_graph, "canonical_graph_counts_after": after_graph, "canonical_graph_writes_performed": before_graph != after_graph})
            if before_graph != after_graph:
                blockers.append("canonical_graph_counts_changed")
        if blockers:
            conn.rollback()
        else:
            conn.commit()
    return result, blockers


def run_job(job_kind: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    if job_kind == "momentary_flow":
        from absurd_momentary_flow import run_momentary_flow
        result = run_momentary_flow(payload)
        ok = result.get("outcome") == "succeeded"
        hygiene_ok, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
        if not hygiene_ok:
            result["hygiene"] = hygiene
            return False, result, hygiene.get("error", "job_result_hygiene_failed")
        return ok, result, "" if ok else "momentary_flow_failed"
    if job_kind == "noop":
        result = {"ok": True, "message": payload.get("message", "noop"), "outcome": "succeeded"}
        ok, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
        if not ok:
            result["hygiene"] = hygiene
            return False, result, hygiene.get("error", "job_result_hygiene_failed")
        return True, result, ""
    if job_kind == "status_ledger_check":
        proc = subprocess.run([str(PY), str(ROOT / "scripts" / "lucidota_status_ledger.py"), "--check"], cwd=ROOT, text=True, capture_output=True, timeout=60)
        ok = proc.returncode == 0
        result = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "outcome": "succeeded" if ok else "failed",
        }
        hygiene_ok, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
        if not hygiene_ok:
            result["hygiene"] = hygiene
            return False, result, hygiene.get("error", "job_result_hygiene_failed")
        return ok, result, "" if ok else proc.stderr[-1000:]
    if job_kind == "external_command":
        command = payload.get("command")
        if not isinstance(command, list) or len(command) < 2:
            result = {"error": "external_command_requires_command_list", "outcome": "failed"}
            ok, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
            if not ok:
                result["hygiene"] = hygiene
                return False, result, "external_command_requires_command_list"
            return False, result, "external_command_requires_command_list"
        if str(command[0]) not in {"python3", "/usr/bin/python3", str(PY)} or str(command[1]) not in ALLOWED_EXTERNAL_COMMANDS:
            result = {"error": "external_command_not_allowlisted", "command": command[:2], "outcome": "failed"}
            ok, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
            if not ok:
                result["hygiene"] = hygiene
            return False, result, "external_command_not_allowlisted"
        script = ROOT / str(command[1])
        if not script.exists() or not script.is_file():
            result = {"error": "external_command_script_missing", "script": str(command[1]), "outcome": "failed"}
            ok, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
            if not ok:
                result["hygiene"] = hygiene
            return False, result, "external_command_script_missing"
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=int(payload.get("timeout_seconds", 180)))
        ok = proc.returncode == 0
        result = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "command": shlex.join(str(x) for x in command),
            "outcome": "succeeded" if ok else "failed",
        }
        hygiene_ok, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
        if not hygiene_ok:
            result["hygiene"] = hygiene
            return False, result, hygiene.get("error", "job_result_hygiene_failed")
        return ok, result, "" if ok else proc.stderr[-1000:]
    result = {"error": "unsupported_job_kind", "job_kind": job_kind, "outcome": "failed"}
    _, hygiene = gate_worker_payload_hygiene(result, queue_name="absurd_queue_spine", job_kind=job_kind, worker_key="absurd_queue_spine")
    result["hygiene"] = hygiene
    return False, result, "unsupported_job_kind"


def worker_once(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url = db_url(args)
    blockers: list[str] = []
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {"database_url": redacted(url), "queue": args.queue, "worker_id": worker_id, "execute_performed": False, "job_processed": False}
    if not execute:
        with connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, status::text
                    FROM lucidota_control.absurd_queue_job
                    WHERE queue_name=%s AND status='queued' AND run_after <= now()
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                """, (args.queue,))
                row = cur.fetchone()
                result["would_process"] = dict(zip(["job_uuid","workflow_name","job_kind","idempotency_key","status"], row)) if row else None
        return result, blockers
    with connect(url) as conn:
        with conn.cursor() as cur:
            before_graph = counts(cur, CANONICAL_GRAPH_TABLES)
            cur.execute("""
                SELECT job_uuid, workflow_name, job_kind, idempotency_key, payload, attempt_count, max_attempts
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after <= now()
                ORDER BY priority ASC, created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            """, (args.queue,))
            row = cur.fetchone()
            if not row:
                result["no_job_available"] = True
                return result, blockers
            job_uuid, workflow_name, job_kind, idempotency_key, payload, attempt_count, max_attempts = row
            cur.execute("""
                UPDATE lucidota_control.absurd_queue_job
                SET status='running', leased_by=%s, lease_expires_at=now() + interval '5 minutes', attempt_count=attempt_count+1, updated_at=now()
                WHERE job_uuid=%s
            """, (worker_id, job_uuid))
            cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail) VALUES (%s,%s,'started',%s::jsonb)", (job_uuid, args.queue, json.dumps({"worker_id":worker_id})))
            ok, job_result, error = run_job(job_kind, payload)
            if ok:
                hygiene_ok, hygiene = gate_worker_payload_hygiene(job_result, queue_name=args.queue, worker_key="absurd_queue_spine", job_kind=job_kind)
                if not hygiene_ok:
                    ok = False
                    error = hygiene.get("error", "job_result_hygiene_failed")
                    job_result.setdefault("hygiene", hygiene)
            if ok:
                cur.execute("""
                    UPDATE lucidota_control.absurd_queue_job
                    SET status='succeeded', result=%s::jsonb, completed_at=now(), updated_at=now(), last_error=''
                    WHERE job_uuid=%s
                """, (json.dumps(job_result), job_uuid))
                cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail) VALUES (%s,%s,'succeeded',%s::jsonb)", (job_uuid, args.queue, json.dumps(job_result)))
                cur.execute("""
                    INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                    VALUES (%s, %s, 'absurd_queue_spine', 'succeeded', 'absurd_queue_spine', %s::jsonb)
                    RETURNING event_id::text
                """, (workflow_name, str(job_uuid), json.dumps({"job_uuid":str(job_uuid), "queue": args.queue, "job_kind": job_kind, "result": job_result})))
                event_id = cur.fetchone()[0]
                result.update({"workflow_event_id": event_id})
            else:
                final_attempt = int(attempt_count) + 1 >= int(max_attempts)
                new_status = 'dead_lettered' if final_attempt else 'failed'
                cur.execute("""
                    UPDATE lucidota_control.absurd_queue_job
                    SET status=%s, result=%s::jsonb, updated_at=now(), last_error=%s
                    WHERE job_uuid=%s
                """, (new_status, json.dumps(job_result), error, job_uuid))
                cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail) VALUES (%s,%s,%s,%s::jsonb)", (job_uuid, args.queue, 'dead_lettered' if final_attempt else 'failed', json.dumps({"error":error, "result":job_result})))
                if final_attempt:
                    cur.execute("""
                        INSERT INTO lucidota_control.absurd_queue_dead_letter
                          (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, error_kind, error_message, attempt_count, payload_sha256, context)
                        VALUES (%s,%s,%s,%s,%s,'job_failed',%s,%s,%s,%s::jsonb)
                        ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET
                          error_message=EXCLUDED.error_message,
                          attempt_count=EXCLUDED.attempt_count,
                          last_seen_at=now(),
                          context=EXCLUDED.context
                    """, (job_uuid, args.queue, workflow_name, job_kind, idempotency_key, error, int(attempt_count)+1, sha256_obj(payload), json.dumps(job_result)))
            after_graph = counts(cur, CANONICAL_GRAPH_TABLES)
            result.update({"execute_performed": True, "job_processed": True, "job_uuid": str(job_uuid), "workflow_name": workflow_name, "job_kind": job_kind, "status": "succeeded" if ok else new_status, "canonical_graph_counts_before": before_graph, "canonical_graph_counts_after": after_graph, "canonical_graph_writes_performed": before_graph != after_graph})
            if before_graph != after_graph:
                blockers.append("canonical_graph_counts_changed")
        if blockers:
            conn.rollback()
        else:
            conn.commit()
    return result, blockers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state")
    ap.add_argument("--action", choices=["audit", "init-schema", "enqueue", "worker-once"], required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--queue", default="control")
    ap.add_argument("--workflow", default="status-ledger-check")
    ap.add_argument("--job-kind", default="status_ledger_check")
    ap.add_argument("--payload-json", default='{}')
    ap.add_argument("--idempotency-key")
    ap.add_argument("--priority", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--worker-id")
    args = ap.parse_args()
    execute = bool(args.execute)
    blockers: list[str] = []
    try:
        if args.action == "init-schema":
            action_result, blockers = apply_schema(args, execute)
        elif args.action == "audit":
            action_result, blockers = audit(args)
        elif args.action == "enqueue":
            action_result, blockers = enqueue(args, execute)
        else:
            action_result, blockers = worker_once(args, execute)
    except Exception as exc:
        action_result = {}
        blockers = [f"exception:{exc}"]
    report = {
        "schema": "lucidota.absurd_queue_spine.report.v1",
        "generated_at": now_iso(),
        "action": args.action,
        "mode": "execute" if execute else "dry_run",
        "execute_requested": execute,
        "action_result": action_result,
        "db_writes_performed": bool(action_result.get("execute_performed")) if isinstance(action_result, dict) else False,
        "canonical_graph_writes_performed": bool(action_result.get("canonical_graph_writes_performed")) if isinstance(action_result, dict) else False,
        "blockers": blockers,
    }
    write_report(args.action, report)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
