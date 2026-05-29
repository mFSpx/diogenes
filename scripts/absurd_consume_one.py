#!/usr/bin/env python3
"""Consume one ABSURD queue job with kernel-authorization enforcement.

This is the ABSURD/Postgres successor to the legacy DBOS consume-one worker.
It claims exactly one queued row, validates kernel authorization before handler
execution, writes job/event/dead-letter receipts, and never mutates canonical
graph tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from absurd_worker_contracts import validate_worker_contract
from spine_kernel_authorization import validate_job_kernel_authorization

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "absurd"

ALLOWED_EXTERNAL_COMMANDS = {
    "scripts/chrono_queue_event_bridge.py",
    "scripts/document_claim_packet_worker.py",
    "scripts/tracer_claim_packet_bridge_dry_run.py",
    "scripts/phase05_design_atom_extractor.py",
    "scripts/simplemem_candidate_index.py",
    "scripts/graph_promotion_gate.py",
    "scripts/spine_krampus_worker.py",
    "scripts/absurd_river_worker.py",
    "scripts/lucidota_model_turbine_overseer.py",
    "scripts/goal_agent_packet.py",
    "scripts/goal_dev_control.py",
    "scripts/goal_model_fabric_control.py",
    "scripts/groq_goal_delegate.py",
    "scripts/goal_model_fabric_orchestrate.py",
    "scripts/model_runner_cli.py",
    "scripts/language_router.py",
    "scripts/lucidota_usecase_proof.py",
}


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def db(args: argparse.Namespace) -> str:
    return (
        args.database_url
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_report(name: str, report: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"absurd_consume_one_{name}_{stamp()}.json"
    report.setdefault("generated_at", now())
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def handle(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    handler = str(payload.get("handler", "noop"))
    if handler == "noop":
        return True, {"message": payload.get("message", "noop")}

    if handler == "status_ledger_check":
        proc = subprocess.run(
            [sys.executable, "scripts/lucidota_status_ledger.py", "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        return proc.returncode == 0, {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }

    if handler == "external_command":
        command = payload.get("command")
        if not isinstance(command, list) or len(command) < 2:
            return False, {"error": "external_command_requires_command_list"}
        if command[0] not in {"python3", "/usr/bin/python3", sys.executable} or str(command[1]) not in ALLOWED_EXTERNAL_COMMANDS:
            return False, {"error": "external_command_not_allowlisted", "command": command[:2]}
        script = ROOT / str(command[1])
        if not script.exists() or not script.is_file():
            return False, {"error": "external_command_script_missing", "script": str(command[1])}
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=int(payload.get("timeout_seconds", 180)),
        )
        return proc.returncode == 0, {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
            "command": shlex.join(str(x) for x in command),
        }

    return False, {"error": "unsupported_handler", "handler": handler}


def consume(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "action": "consume_one",
        "queue_name": args.queue_name,
        "execute_performed": bool(args.execute),
        "job_processed": False,
        "blockers": [],
    }

    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if not args.execute:
                cur.execute(
                    """
                    SELECT job_uuid::text,idempotency_key,payload,job_kind,status::text
                    FROM lucidota_control.absurd_queue_job
                    WHERE queue_name=%s AND status='queued' AND run_after<=now()
                    ORDER BY priority,created_at
                    LIMIT 1
                    """,
                    (args.queue_name,),
                )
                row = cur.fetchone()
                report["would_process"] = dict(row) if row else None
                if row:
                    payload = dict(row["payload"] or {})
                    contract = validate_worker_contract(
                        cur,
                        queue_name=args.queue_name,
                        job_kind=str(payload.get("job_kind") or row.get("job_kind") or ""),
                    )
                    report["worker_contract"] = contract.as_result()
                write_report("dry_run", report)
                return 0

            cur.execute("SET LOCAL lucidota.actor_role='worker'")
            cur.execute(
                """
                SELECT job_uuid::text,idempotency_key,payload,attempt_count,max_attempts,job_kind,workflow_name
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after<=now()
                ORDER BY priority,created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (args.queue_name,),
            )
            row = cur.fetchone()
            if not row:
                report["blockers"].append("no_queued_job")
                write_report("execute", report)
                return 0

            job_uuid = row["job_uuid"]
            payload = dict(row["payload"] or {})
            contract = validate_worker_contract(
                cur,
                queue_name=args.queue_name,
                job_kind=str(row["job_kind"]),
            )
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status='running',
                    locked_by=%s,
                    leased_by=%s,
                    locked_at=now(),
                    lease_expires_at=now()+interval '5 minutes',
                    last_heartbeat_at=now(),
                    attempt_count=attempt_count+1
                WHERE job_uuid=%s::uuid
                """,
                (args.worker_id, args.worker_id, job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event(job_uuid,queue_name,event_kind,event_source,detail)
                VALUES (%s::uuid,%s,'started','absurd_consume_one',%s::jsonb)
                """,
                (job_uuid, args.queue_name, json.dumps({"worker_id": args.worker_id})),
            )

            if not contract.ok:
                ok = False
                result = {"error": "worker_contract_rejected", "worker_contract": contract.as_result()}
                error_kind = contract.error_kind or "worker_contract_error"
                error_message = contract.error_message or "worker contract rejected job"
            else:
                auth = validate_job_kernel_authorization(
                    queue_name=args.queue_name,
                    job_kind=str(row["job_kind"]),
                    payload=payload,
                )
                if not auth.ok:
                    ok = False
                    result = {"error": "kernel_authorization_rejected", "kernel_authorization": auth.as_result()}
                    error_kind = auth.error_kind or "kernel_authorization_error"
                    error_message = auth.error_message or "kernel authorization rejected job"
                else:
                    ok, result = handle(payload)
                    error_kind = "" if ok else "handler_error"
                    error_message = "" if ok else json.dumps(result, sort_keys=True, default=str)
            status = "succeeded" if ok else ("dead_lettered" if int(row["attempt_count"]) + 1 >= int(row["max_attempts"]) else "failed")
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status=%s,
                    result=%s::jsonb,
                    last_heartbeat_at=now(),
                    error_kind=%s,
                    error_message=%s,
                    last_error=%s,
                    completed_at=CASE WHEN %s='succeeded' THEN now() ELSE completed_at END
                WHERE job_uuid=%s::uuid
                """,
                (status, json.dumps(result, default=str), error_kind, error_message, error_message, status, job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event(job_uuid,queue_name,event_kind,event_source,detail)
                VALUES (%s::uuid,%s,%s,'absurd_consume_one',%s::jsonb)
                """,
                (job_uuid, args.queue_name, status if status in {"succeeded", "failed", "dead_lettered"} else "failed", json.dumps(result, default=str)),
            )
            if status == "dead_lettered":
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_dead_letter
                      (job_uuid,queue_name,workflow_name,job_kind,idempotency_key,error_kind,error_message,attempt_count,payload_sha256,context)
                    VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET
                      error_kind=EXCLUDED.error_kind,
                      error_message=EXCLUDED.error_message,
                      attempt_count=EXCLUDED.attempt_count,
                      last_seen_at=now(),
                      context=EXCLUDED.context
                    """,
                    (
                        job_uuid,
                        args.queue_name,
                        row["workflow_name"],
                        row["job_kind"],
                        row["idempotency_key"],
                        error_kind or "handler_error",
                        error_message,
                        int(row["attempt_count"]) + 1,
                        sha256_obj(payload),
                        json.dumps(result, default=str),
                    ),
                )
        conn.commit()

    report.update({"job_processed": True, "job_uuid": job_uuid, "status": status, "result": result, "worker_contract": contract.as_result()})
    write_report("execute", report)
    print(f"JOB_UUID={job_uuid}")
    print(f"STATUS={status}")
    return 0 if status == "succeeded" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Consume one ABSURD queue job with kernel authorization.")
    parser.add_argument("--database-url")
    parser.add_argument("--queue-name", default="boring_beast")
    parser.add_argument("--worker-id", default="absurd-consume-one")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return consume(args)


if __name__ == "__main__":
    raise SystemExit(main())
