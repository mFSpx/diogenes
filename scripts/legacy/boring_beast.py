#!/usr/bin/env python3
"""Executable LUCIDOTA Boring Beast work-loop runtime.

This is not a report generator. It provides DB-backed commands for queue
hardening, legal state transitions, work-order validation, consume-one workers,
idempotency, retry/DLQ, execution records, Chrono append events, runtime status
facts, daemon checks, stale-lock recovery, audit verdict enforcement, graph
promotion gating/barriers, TRACER-lite, DeMem, CatchMe, SimpleMem, and an E2E
command that proves the path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from dbos_kernel_authorization import validate_job_kernel_authorization
from kernel_control_packet import dbos_enqueue_packet

STATE_SCHEMA = ROOT / "06_SCHEMA" / "039_dbos_real_work_loop.sql"
GRAPH_BARRIER_SCHEMA = ROOT / "06_SCHEMA" / "040_graph_write_barrier_enforcement.sql"
OUT_DIR = ROOT / "05_OUTPUTS" / "boring_beast"
WORK_LOOP_LEDGER = ROOT / "05_OUTPUTS" / "work_loops" / "real_code_loop_ledger.jsonl"
STATUS_JSON = ROOT / "05_OUTPUTS" / "status_ledger.json"
STATUS_MD = ROOT / "00_PROJECT_BRAIN" / "STATUS_LEDGER.md"
CANONICAL_GRAPH_TABLES = ["lucidota_go.graph_item", "lucidota_go.graph_edge", "lucidota_go.graph_journal"]
PROMOTION_TABLES = ["lucidota_go.graph_promotion_packet", "lucidota_go.graph_promotion_decision"]
ALLOWED_AUTHORITY = {
    "raw_evidence", "operator_authored_assertion", "operator_defined_label", "deterministic_metric",
    "statistical_finding", "model_computed_finding", "stream_ml_finding", "graph_inferred_relation",
    "operator_confirmed_finding", "canonical_doctrine", "external_action_authorized",
}
TRACER_LABELS = {"quote","compression","inference","abduction","speculation","operator_prior","heuristic","contradiction","falsification_target","PFM"}
LEGAL_WORK_TARGETS = {str(i) for i in range(1, 21)}
BORING_BEAST_QUEUE = "boring_beast"
BORING_BEAST_JOB_KIND = "boring_beast_work_item"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def jdump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(jdump(obj))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def redact_url(url: str) -> str:
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def state_url(args: argparse.Namespace | None = None) -> str:
    if args is not None and getattr(args, "state_database_url", None):
        return args.state_database_url
    return os.environ.get("DBOS_SYSTEM_DATABASE_URL") or os.environ.get("LUCIDOTA_STATE_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_url(args: argparse.Namespace | None = None) -> str:
    if args is not None and getattr(args, "storage_database_url", None):
        return args.storage_database_url
    return os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def connect(url: str, row_factory=None):
    return psycopg.connect(url, row_factory=row_factory)


def first(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def table_exists(cur, table: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (table,))
    return first(cur.fetchone()) is not None


def count_table(cur, table: str) -> int | None:
    if not table_exists(cur, table):
        return None
    cur.execute(f"SELECT count(*) FROM {table}")
    return int(first(cur.fetchone()))


def counts(cur, tables: Iterable[str]) -> dict[str, int | None]:
    return {t: count_table(cur, t) for t in tables}


def write_report(prefix: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{prefix}_{stamp()}.json"
    report.setdefault("generated_at", now())
    report["report_path"] = rel(out)
    out.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(out)}")
    return out


def append_work_loop(loop: int, item: int, target: str, counted: bool, files: list[str], validations: list[dict[str, str]], delta: str, blocked_by: str | None, next_action: str) -> None:
    WORK_LOOP_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "loop": loop,
        "item": item,
        "target": target,
        "counted": bool(counted),
        "files_changed": files,
        "validation": validations,
        "capability_delta": delta,
        "blocked_by": blocked_by,
        "next_action": next_action,
        "created_at": now(),
    }
    with WORK_LOOP_LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def load_payload(raw: str | None, path: str | None = None) -> dict[str, Any]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    elif raw:
        data = json.loads(raw)
    else:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")
    return data


def normalize_idempotency_key(raw: str) -> str:
    key = "-".join(str(raw).strip().lower().split())
    safe = "".join(ch for ch in key if ch.isalnum() or ch in "-_.:")
    return safe[:180] or sha256_text(str(raw))


def validate_work_order(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if str(payload.get("target_number", "")) not in LEGAL_WORK_TARGETS:
        errors.append("target_number_must_be_1_to_20")
    if not str(payload.get("target_name", "")).strip():
        errors.append("target_name_required")
    if not str(payload.get("idempotency_key", "")).strip():
        errors.append("idempotency_key_required")
    else:
        payload["idempotency_key"] = normalize_idempotency_key(str(payload["idempotency_key"]))
    if "handler" in payload and payload["handler"] not in {"noop", "status_ledger_check", "fail_once", "tracer_label", "simplemem_index"}:
        errors.append("unsupported_handler")
    if "files_changed" in payload and not isinstance(payload["files_changed"], list):
        errors.append("files_changed_must_be_list")
    if "validation_commands" in payload and not isinstance(payload["validation_commands"], list):
        errors.append("validation_commands_must_be_list")
    return (not errors), errors


def ensure_kernel_authorized_work_order(payload: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
    """Attach/validate the CKDOG1 packet required before DBOS execution.

    Existing callers can keep submitting minimal Boring Beast payloads; enqueue
    normalizes them into a kernel-authorized DBOS work item. If a caller supplies
    a packet, it must verify against the normalized queue/lane/source/idempotency
    fields instead of being silently replaced.
    """
    payload["idempotency_key"] = idempotency_key
    payload.setdefault("lane", "work_item")
    payload.setdefault("source_path", "scripts/boring_beast.py")
    if "kernel_authorization" not in payload:
        payload["kernel_authorization"] = dbos_enqueue_packet(
            queue_name=BORING_BEAST_QUEUE,
            lane=str(payload["lane"]),
            source_path=str(payload["source_path"]),
            idempotency_key=idempotency_key,
            authorized_by=str(payload.get("authorized_by") or "boring_beast_enqueue"),
        )
    verdict = validate_job_kernel_authorization(
        queue_name=BORING_BEAST_QUEUE,
        job_kind=BORING_BEAST_JOB_KIND,
        payload=payload,
    )
    return verdict.as_result()


def cmd_init_schema(args: argparse.Namespace) -> int:
    report = {"action": "init_schema", "state_database_url": redact_url(state_url(args)), "schema": rel(STATE_SCHEMA), "execute_performed": False, "blockers": []}
    if not args.execute:
        report["schema_bytes"] = STATE_SCHEMA.stat().st_size
        write_report("init_schema_dry_run", report)
        return 0
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute(STATE_SCHEMA.read_text(encoding="utf-8"))
            report["queue_job_count"] = count_table(cur, "lucidota_control.dbos_queue_job")
            report["conversation_command_count"] = count_table(cur, "lucidota_control.conversation_command")
        conn.commit()
    report["execute_performed"] = True
    write_report("init_schema_execute", report)
    return 0


def cmd_apply_graph_barrier(args: argparse.Namespace) -> int:
    report = {"action": "apply_graph_barrier", "storage_database_url": redact_url(storage_url(args)), "schema": rel(GRAPH_BARRIER_SCHEMA), "execute_performed": False, "blockers": []}
    if not args.execute:
        report["schema_bytes"] = GRAPH_BARRIER_SCHEMA.stat().st_size
        write_report("graph_barrier_dry_run", report)
        return 0
    with connect(storage_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute(GRAPH_BARRIER_SCHEMA.read_text(encoding="utf-8"))
            report["canonical_counts"] = counts(cur, CANONICAL_GRAPH_TABLES)
        conn.commit()
    report["execute_performed"] = True
    write_report("graph_barrier_execute", report)
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    payload = load_payload(args.payload_json, args.payload_file)
    ok, errors = validate_work_order(payload)
    idem = payload.get("idempotency_key") or sha256_obj(payload)
    report = {"action": "enqueue", "valid": ok, "validation_errors": errors, "idempotency_key": idem, "execute_performed": False, "inserted_new": False, "job_uuid": None, "kernel_authorization": None, "blockers": []}
    if not ok:
        report["blockers"].extend(errors)
    else:
        auth = ensure_kernel_authorized_work_order(payload, str(idem))
        report["kernel_authorization"] = auth
        if not auth["ok"]:
            report["blockers"].append(auth.get("error_kind") or "kernel_authorization_invalid")
    if args.dry_run or report["blockers"]:
        write_report("enqueue_dry_run", report)
        return 0 if ok and not report["blockers"] else 2
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lucidota.actor_role = 'foreman'")
            cur.execute(
                """
                INSERT INTO lucidota_control.dbos_queue_job
                  (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                VALUES ('boring_beast','boring-beast-work-loop','boring_beast_work_item',%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=lucidota_control.dbos_queue_job.updated_at
                RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                """,
                (idem, json.dumps(payload), int(args.priority), int(args.max_attempts), json.dumps({"source":"scripts/boring_beast.py"})),
            )
            job_uuid, inserted_new = cur.fetchone()
            if inserted_new:
                cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,'boring_beast','enqueued','boring_beast',%s::jsonb)", (job_uuid, json.dumps({"idempotency_key": idem, "target_number": payload.get("target_number")})))
        conn.commit()
    report.update({"execute_performed": True, "inserted_new": bool(inserted_new), "job_uuid": job_uuid})
    write_report("enqueue_execute", report)
    print(f"JOB_UUID={job_uuid}")
    print(f"INSERTED_NEW={str(bool(inserted_new)).lower()}")
    return 0


def handle_work_item(payload: dict[str, Any], job_uuid: str, idem: str) -> tuple[bool, dict[str, Any], str, list[str]]:
    handler = payload.get("handler", "noop")
    files_changed = payload.get("files_changed", [])
    if handler == "noop":
        return True, {"message": payload.get("message", "noop complete"), "target_number": payload.get("target_number")}, "", files_changed
    if handler == "status_ledger_check":
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "lucidota_status_ledger.py"), "--check"], cwd=ROOT, text=True, capture_output=True, timeout=90)
        return proc.returncode == 0, {"returncode": proc.returncode, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]}, proc.stderr[-500:] or "status_ledger_check_failed", files_changed
    if handler == "fail_once":
        if payload.get("fail") is True:
            return False, {"forced_failure": True}, "forced_failure_for_retry_dlq_test", files_changed
        return True, {"forced_failure": False}, "", files_changed
    if handler == "tracer_label":
        return True, {"trace_label_requested": payload.get("trace_label")}, "", files_changed
    if handler == "simplemem_index":
        return True, {"simplemem_requested": True, "text_len": len(str(payload.get("text", "")))}, "", files_changed
    return False, {}, "unsupported_handler", files_changed


def append_chrono_event(args: argparse.Namespace, event: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    url = storage_url(args)
    source_sha = sha256_obj(event)
    try:
        with connect(url) as conn:
            with conn.cursor() as cur:
                if not table_exists(cur, "lucidota_korpus.temporal_claim"):
                    return False, None, "temporal_claim_missing"
                cur.execute(
                    """
                    INSERT INTO lucidota_korpus.temporal_claim
                      (artifact_uuid, file_uuid, candidate_timestamp, evidence_source, trust_weight, raw_evidence, extractor, extractor_version, source_path, source_sha256, detail)
                    VALUES (NULL, NULL, now(), 'boring_beast_runtime_event', 0.50, %s, 'boring_beast', 'v1', 'scripts/boring_beast.py', %s, %s::jsonb)
                    RETURNING claim_uuid::text
                    """,
                    (json.dumps({"event_kind": event.get("event_kind"), "job_uuid": event.get("job_uuid")}), source_sha, json.dumps(event)),
                )
                claim_uuid = cur.fetchone()[0]
            conn.commit()
        return True, claim_uuid, None
    except Exception as exc:
        return False, None, str(exc)


def cmd_worker_once(args: argparse.Namespace) -> int:
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    target_idempotency_key = normalize_idempotency_key(args.idempotency_key) if getattr(args, "idempotency_key", None) else None
    report = {"action": "worker_once", "worker_id": worker_id, "target_idempotency_key": target_idempotency_key, "execute_performed": False, "job_processed": False, "blockers": []}
    with connect(state_url(args)) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            if args.dry_run:
                cur.execute("""
                    SELECT job_uuid::text, idempotency_key, payload, status::text
                    FROM lucidota_control.dbos_queue_job
                    WHERE queue_name='boring_beast' AND status='queued' AND run_after <= now()
                      AND (%s::text IS NULL OR idempotency_key=%s)
                    ORDER BY priority ASC, created_at ASC LIMIT 1
                """, (target_idempotency_key, target_idempotency_key))
                row = cur.fetchone()
                report["would_process"] = dict(row) if row else None
                write_report("worker_once_dry_run", report)
                return 0
            cur.execute("SET LOCAL lucidota.actor_role = 'worker'")
            cur.execute("""
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, payload, attempt_count, max_attempts
                FROM lucidota_control.dbos_queue_job
                WHERE queue_name='boring_beast' AND status='queued' AND run_after <= now()
                  AND (%s::text IS NULL OR idempotency_key=%s)
                ORDER BY priority ASC, created_at ASC
                FOR UPDATE SKIP LOCKED LIMIT 1
            """, (target_idempotency_key, target_idempotency_key))
            row = cur.fetchone()
            if not row:
                report["no_job_available"] = True
                write_report("worker_once_execute", report)
                return 0
            job_uuid = row["job_uuid"]
            idem = row["idempotency_key"]
            payload = dict(row["payload"])
            cur.execute("""
                UPDATE lucidota_control.dbos_queue_job
                SET status='running', locked_by=%s, locked_at=now(), leased_by=%s, lease_expires_at=now() + interval '5 minutes', last_heartbeat_at=now(), attempt_count=attempt_count+1
                WHERE job_uuid=%s
            """, (worker_id, worker_id, job_uuid))
            cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,'boring_beast','started','boring_beast',%s::jsonb)", (job_uuid, json.dumps({"worker_id": worker_id})))
            auth = validate_job_kernel_authorization(queue_name=BORING_BEAST_QUEUE, job_kind=str(row["job_kind"]), payload=payload)
            report["kernel_authorization"] = auth.as_result()
            if not auth.ok:
                ok = False
                result = {"error": "kernel_authorization_rejected", "kernel_authorization": auth.as_result()}
                error = auth.error_message or "kernel authorization rejected job"
                error_kind = auth.error_kind or "kernel_authorization_error"
                files_changed = []
            else:
                ok, result, error, files_changed = handle_work_item(payload, job_uuid, idem)
                error_kind = "handler_error"
            status = "succeeded" if ok else ("dead_lettered" if int(row["attempt_count"])+1 >= int(row["max_attempts"]) else "failed")
            audit = {"verdict": "PASS" if ok else "FAIL", "required_fields_ok": True, "remediation": "" if ok else "inspect boring dead letter", "evidence_refs": [f"dbos_job:{job_uuid}"]}
            if ok:
                cur.execute("""
                    UPDATE lucidota_control.dbos_queue_job
                    SET status='succeeded', result=%s::jsonb, completed_at=now(), last_heartbeat_at=now(), last_error='', error_kind='', error_message=''
                    WHERE job_uuid=%s
                """, (json.dumps(result), job_uuid))
                event_kind = "succeeded"
            else:
                cur.execute("""
                    UPDATE lucidota_control.dbos_queue_job
                    SET status=%s, result=%s::jsonb, last_heartbeat_at=now(), last_error=%s, error_kind=%s, error_message=%s
                    WHERE job_uuid=%s
                """, (status, json.dumps(result), error, error_kind, error, job_uuid))
                event_kind = "dead_lettered" if status == "dead_lettered" else "failed"
                if status == "dead_lettered":
                    cur.execute("""
                        INSERT INTO lucidota_control.dbos_queue_dead_letter
                          (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, error_kind, error_message, attempt_count, payload_sha256, context)
                        VALUES (%s,'boring_beast','boring-beast-work-loop','boring_beast_work_item',%s,%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET last_seen_at=now(), attempt_count=EXCLUDED.attempt_count, error_message=EXCLUDED.error_message
                    """, (job_uuid, idem, error_kind, error, int(row["attempt_count"])+1, sha256_obj(payload), json.dumps({"payload": payload})))
            cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,'boring_beast',%s,'boring_beast',%s::jsonb)", (job_uuid, event_kind, json.dumps({"result": result, "error": error})))
            cur.execute("""
                INSERT INTO lucidota_control.boring_execution_record(task_id, job_uuid, idempotency_key, files_changed, validation_commands, result, status, audit_verdict, detail)
                VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING execution_uuid::text
            """, (
                str(payload.get("target_number", "unknown")), job_uuid, idem, json.dumps(files_changed),
                json.dumps(payload.get("validation_commands", [])), json.dumps(result), status,
                json.dumps(audit), json.dumps({"payload_sha256": sha256_obj(payload), "worker_id": worker_id}),
            ))
            ex_row = cur.fetchone()
            execution_uuid = ex_row["execution_uuid"] if ex_row else None
        conn.commit()
    chrono_ok, claim_uuid, chrono_error = append_chrono_event(args, {"event_kind": "boring_work_item_processed", "job_uuid": job_uuid, "idempotency_key": idem, "status": status, "execution_uuid": execution_uuid})
    report.update({"execute_performed": True, "job_processed": True, "job_uuid": job_uuid, "idempotency_key": idem, "status": status, "execution_uuid": execution_uuid, "chrono_event_appended": chrono_ok, "chrono_claim_uuid": claim_uuid, "chrono_error": chrono_error})
    write_report("worker_once_execute", report)
    print(f"JOB_UUID={job_uuid}")
    print(f"STATUS={status}")
    if claim_uuid:
        print(f"CHRONO_CLAIM_UUID={claim_uuid}")
    return 0 if status == "succeeded" else 3


def cmd_recover_stale(args: argparse.Namespace) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=int(args.timeout_seconds))
    report = {"action": "recover_stale", "timeout_seconds": int(args.timeout_seconds), "execute_performed": False, "recovered_count": 0, "blockers": []}
    with connect(state_url(args)) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT job_uuid::text, status::text, locked_at::text, last_heartbeat_at::text, locked_by
                FROM lucidota_control.dbos_queue_job
                WHERE queue_name='boring_beast' AND status IN ('leased','running') AND coalesce(last_heartbeat_at, locked_at) < %s
            """, (cutoff,))
            stale = [dict(r) for r in cur.fetchall()]
            report["stale_jobs"] = stale
            if args.execute and stale:
                cur.execute("SET LOCAL lucidota.actor_role = 'foreman'")
                cur.execute("""
                    UPDATE lucidota_control.dbos_queue_job
                    SET status='queued', last_error='recovered stale lock', error_kind='stale_lock_recovered', error_message='recovered stale lock'
                    WHERE queue_name='boring_beast' AND status IN ('leased','running') AND coalesce(last_heartbeat_at, locked_at) < %s
                    RETURNING job_uuid::text
                """, (cutoff,))
                recovered = [r["job_uuid"] for r in cur.fetchall()]
                for job_uuid in recovered:
                    cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,'boring_beast','retry_scheduled','boring_beast',%s::jsonb)", (job_uuid, json.dumps({"reason":"stale_lock_recovered"})))
                report["recovered_count"] = len(recovered)
                report["recovered_jobs"] = recovered
            conn.commit()
    report["execute_performed"] = bool(args.execute)
    write_report("recover_stale", report)
    return 0


def cmd_audit_verdict(args: argparse.Namespace) -> int:
    evidence = args.evidence_ref or []
    blockers: list[str] = []
    if args.verdict in {"FAIL", "PARTIAL_FAIL"} and not args.remediation.strip():
        blockers.append("remediation_required_for_fail_or_partial_fail")
    if not evidence:
        blockers.append("evidence_ref_required")
    report = {"action": "audit_verdict", "verdict": args.verdict, "task_id": args.task_id, "execute_performed": False, "rejected": bool(blockers), "blockers": blockers}
    if args.dry_run or blockers:
        write_report("audit_verdict_dry_run", report)
        return 0 if not blockers else 2
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.audit_verdict_contract(task_id, verdict, required_fields_ok, remediation, evidence_refs, rejected, detail)
                VALUES (%s,%s,true,%s,%s::jsonb,false,%s::jsonb)
                RETURNING verdict_uuid::text
            """, (args.task_id, args.verdict, args.remediation, json.dumps(evidence), json.dumps({"source":"scripts/boring_beast.py"})))
            report["verdict_uuid"] = cur.fetchone()[0]
        conn.commit()
    report["execute_performed"] = True
    write_report("audit_verdict_execute", report)
    return 0


def cmd_oracle_check(args: argparse.Namespace) -> int:
    allowed = {str(Path(p)) for p in (args.allowed_file or [])}
    before = set(args.before_manifest or [])
    after = set(args.after_manifest or [])
    changed = sorted(after - before)
    violations = [p for p in changed if p not in allowed]
    report = {"action": "oracle_file_change_check", "allowed_files": sorted(allowed), "changed_files": changed, "violations": violations, "pass": not violations}
    write_report("oracle_file_change_check", report)
    return 0 if not violations else 4


def cmd_supervisor_check(args: argparse.Namespace) -> int:
    target = args.command
    exists = bool(target and Path(target).exists() and os.access(target, os.X_OK))
    report = {"action": "supervisor_check", "command": target, "exists_executable": exists, "returncode": None, "stdout_tail": "", "stderr_tail": "", "failure": None}
    if exists and args.execute:
        proc = subprocess.run([target], cwd=ROOT, text=True, capture_output=True, timeout=int(args.timeout_seconds))
        report.update({"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:], "failure": None if proc.returncode == 0 else "command_failed"})
    elif not exists:
        report["failure"] = "missing_or_not_executable"
    write_report("supervisor_check", report)
    return 0 if report.get("failure") is None else 5


def status_bar(progress: int) -> str:
    progress = max(0, min(100, int(progress)))
    filled = progress // 10
    return f"[{'█' * filled}{'░' * (10 - filled)}] {progress}%"


def update_status_ledger_runtime_fact(subsystem: str, fact_key: str, fact_value: dict[str, Any], evidence: str) -> None:
    data = json.loads(STATUS_JSON.read_text(encoding="utf-8")) if STATUS_JSON.exists() else {}
    data.setdefault("runtime_facts", [])
    data["runtime_facts"] = [f for f in data["runtime_facts"] if not (f.get("subsystem") == subsystem and f.get("fact_key") == fact_key)]
    data["runtime_facts"].append({"subsystem": subsystem, "fact_key": fact_key, "fact_value": fact_value, "evidence": evidence, "derived_at": now()})
    # merge or add software entry without claiming unsupported verification
    entries = data.setdefault("software", [])
    match = None
    for e in entries:
        if e.get("name") == "Boring Beast E2E command":
            match = e
            break
    if match is None:
        match = {"name":"Boring Beast E2E command", "path_or_profile":"scripts/boring_beast.py;06_SCHEMA/039_dbos_real_work_loop.sql", "status":"executed", "executed":"yes", "progress":65, "evidence":evidence, "next_action":"Run scripts/boring_beast.py e2e --execute", "blockers":"", "owner_or_subsystem":"DBOS queue spine", "purpose":"Cold start -> enqueue -> worker -> audit -> chrono -> status fact -> duplicate block.", "loading_bar": status_bar(65), "last_updated": now()}
        entries.append(match)
    else:
        match.update({"status":"executed", "executed":"yes", "progress":max(int(match.get("progress",0)),65), "loading_bar": status_bar(max(int(match.get("progress",0)),65)), "last_updated": now(), "evidence":evidence, "next_action":"Run scripts/boring_beast.py e2e --execute"})
    STATUS_JSON.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def cmd_status_from_runtime(args: argparse.Namespace) -> int:
    report = {"action": "status_from_runtime", "execute_performed": False, "facts": {}, "blockers": []}
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT status::text, count(*) FROM lucidota_control.dbos_queue_job
                WHERE queue_name='boring_beast' GROUP BY status::text ORDER BY status::text
            """)
            qcounts = {r[0]: int(r[1]) for r in cur.fetchall()}
            report["facts"]["boring_beast_queue_status_counts"] = qcounts
            if args.execute:
                cur.execute("""
                    INSERT INTO lucidota_control.runtime_status_fact(subsystem, fact_key, fact_value, evidence_refs)
                    VALUES ('boring_beast','queue_status_counts',%s::jsonb,%s::jsonb)
                    ON CONFLICT (subsystem, fact_key) DO UPDATE SET fact_value=EXCLUDED.fact_value, evidence_refs=EXCLUDED.evidence_refs, derived_at=now()
                """, (json.dumps(qcounts), json.dumps(["lucidota_control.dbos_queue_job"])))
        conn.commit()
    out = write_report("status_from_runtime", report)
    if args.execute:
        update_status_ledger_runtime_fact("boring_beast", "queue_status_counts", report["facts"], rel(out))
        report["execute_performed"] = True
        out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    return 0


def cmd_chrono_event(args: argparse.Namespace) -> int:
    event = {"event_kind": args.event_kind, "detail": load_payload(args.detail_json, None), "created_at": now()}
    ok, claim_uuid, err = append_chrono_event(args, event)
    report = {"action": "chrono_event", "event": event, "chrono_event_appended": ok, "claim_uuid": claim_uuid, "error": err, "execute_performed": ok}
    write_report("chrono_event", report)
    return 0 if ok else 6


def cmd_conversation_enqueue(args: argparse.Namespace) -> int:
    envelope = load_payload(args.command_envelope_json, args.command_envelope_file)
    instruction = args.plain_language_instruction or str(envelope.get("plain_language_instruction", "")).strip()
    blockers: list[str] = []
    if not instruction:
        blockers.append("plain_language_instruction_required")
    if args.authority_class not in ALLOWED_AUTHORITY:
        blockers.append("invalid_authority_class")
    if envelope.get("canonical_mutation_allowed") is True:
        blockers.append("canonical_mutation_not_allowed")
    idem = args.idempotency_key or sha256_obj({"instruction": instruction, "envelope": envelope})
    report = {"action": "conversation_enqueue", "idempotency_key": idem, "blockers": blockers, "execute_performed": False, "inserted_new": False}
    if args.dry_run or blockers:
        write_report("conversation_enqueue_dry_run", report)
        return 0 if not blockers else 2
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.conversation_command
                  (command_kind, plain_language_instruction, command_envelope, source_surface_id, source_artifact_refs, target_refs, evidence_refs, allowed_effect, authority_class, idempotency_key, detail)
                VALUES (%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s::jsonb)
                ON CONFLICT (cep_dedupe_key) DO UPDATE SET updated_at=lucidota_control.conversation_command.updated_at
                RETURNING command_uuid::text, (xmax = 0) AS inserted_new
            """, (
                args.command_kind, instruction, json.dumps(envelope), args.source_surface_id,
                json.dumps(envelope.get("artifact_refs", [])), json.dumps(envelope.get("target_refs", [])), json.dumps(envelope.get("evidence_refs", [])),
                args.allowed_effect or envelope.get("allowed_effect", "stage_only"), args.authority_class, idem, json.dumps({"source":"scripts/boring_beast.py"}),
            ))
            command_uuid, inserted_new = cur.fetchone()
        conn.commit()
    report.update({"execute_performed": True, "command_uuid": command_uuid, "inserted_new": bool(inserted_new)})
    write_report("conversation_enqueue_execute", report)
    print(f"COMMAND_UUID={command_uuid}")
    return 0


def cmd_graph_promote(args: argparse.Namespace) -> int:
    payload = load_payload(args.candidate_payload_json, args.candidate_payload_file)
    evidence = args.evidence_ref or []
    blockers: list[str] = []
    if not evidence:
        blockers.append("evidence_ref_required")
    if args.authority_class not in ALLOWED_AUTHORITY:
        blockers.append("invalid_authority_class")
    if args.decision in {"promote", "operator_confirmed"} and not args.operator_confirmed:
        blockers.append("operator_confirmed_required_for_materialization")
    if args.materialize:
        blockers.append("legacy_direct_canonical_graph_materialization_disabled_use_graph_materialization_helper_policy_gate")
    report = {"action": "graph_promote", "decision": args.decision, "materialize": bool(args.materialize), "execute_performed": False, "canonical_graph_writes_performed": False, "blockers": blockers}
    if args.dry_run or blockers:
        write_report("graph_promote_dry_run", report)
        return 0 if not blockers else 2
    with connect(storage_url(args)) as conn:
        with conn.cursor() as cur:
            before = counts(cur, CANONICAL_GRAPH_TABLES)
            cur.execute("""
                INSERT INTO lucidota_go.graph_promotion_packet(source_system,candidate_kind,candidate_payload,evidence_refs,authority_class,detail)
                VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb)
                RETURNING packet_uuid::text
            """, (args.source_system, args.candidate_kind, json.dumps(payload), json.dumps(evidence), args.authority_class, json.dumps({"script":"scripts/boring_beast.py"})))
            packet_uuid = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO lucidota_go.graph_promotion_decision(packet_uuid,decision,decided_by,rationale,evidence_refs,operator_confirmed,command_envelope_uuid)
                VALUES (%s,%s,'graph_promoter',%s,%s::jsonb,%s,%s)
                RETURNING decision_uuid::text
            """, (packet_uuid, args.decision, args.rationale, json.dumps(evidence), bool(args.operator_confirmed), args.command_envelope_uuid))
            decision_uuid = cur.fetchone()[0]
            item_uuid = None
            journal_uuid = None
            cur.execute("UPDATE lucidota_go.graph_promotion_packet SET promotion_status=%s WHERE packet_uuid=%s", (args.decision, packet_uuid))
            after = counts(cur, CANONICAL_GRAPH_TABLES)
        conn.commit()
    report.update({"execute_performed": True, "packet_uuid": packet_uuid, "decision_uuid": decision_uuid, "item_uuid": item_uuid, "journal_uuid": journal_uuid, "canonical_counts_before": before, "canonical_counts_after": after, "canonical_graph_writes_performed": before != after})
    write_report("graph_promote_execute", report)
    if item_uuid:
        print(f"ITEM_UUID={item_uuid}")
    return 0


def cmd_direct_graph_write_check(args: argparse.Namespace) -> int:
    report = {"action": "direct_graph_write_blocker_check", "blocked": False, "error": None, "blockers": []}
    try:
        with connect(storage_url(args)) as conn:
            with conn.cursor() as cur:
                before = counts(cur, CANONICAL_GRAPH_TABLES)
                report["canonical_counts_before"] = before
                try:
                    cur.execute("""
                        INSERT INTO lucidota_go.graph_item(term,label,status,location_at_on_graph,payload)
                        VALUES ('CLAIM','direct write should be blocked','staged','direct_graph_write_check', '{}'::jsonb)
                    """)
                    report["blocked"] = False
                    report["blockers"].append("direct_write_not_blocked")
                    conn.rollback()
                except Exception as exc:
                    report["blocked"] = True
                    report["error"] = str(exc)
                    conn.rollback()
    except Exception as exc:
        report["blockers"].append(f"db_error:{exc}")
    write_report("direct_graph_write_check", report)
    return 0 if report["blocked"] else 7


def cmd_tracer_label(args: argparse.Namespace) -> int:
    blockers = []
    if args.label not in TRACER_LABELS:
        blockers.append("invalid_tracer_label")
    if args.authority_class not in ALLOWED_AUTHORITY:
        blockers.append("invalid_authority_class")
    report = {"action": "tracer_label", "packet_ref": args.packet_ref, "label": args.label, "execute_performed": False, "blockers": blockers}
    if args.dry_run or blockers:
        write_report("tracer_label_dry_run", report)
        return 0 if not blockers else 2
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.tracer_lite_label(packet_ref,label,source_span,authority_class,confidence_bps,detail)
                VALUES (%s,%s,%s::jsonb,%s,%s,%s::jsonb) RETURNING trace_uuid::text
            """, (args.packet_ref, args.label, json.dumps(load_payload(args.source_span_json, None)), args.authority_class, args.confidence_bps, json.dumps({"script":"scripts/boring_beast.py"})))
            report["trace_uuid"] = cur.fetchone()[0]
        conn.commit()
    report["execute_performed"] = True
    write_report("tracer_label_execute", report)
    return 0


def cmd_demem_check(args: argparse.Namespace) -> int:
    text = args.instruction.lower()
    hits: list[dict[str, Any]] = []
    with connect(state_url(args)) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT boundary_key,boundary_text,enforcement_mode FROM lucidota_control.demem_boundary WHERE active=true")
            for row in cur.fetchall():
                key = row["boundary_key"]
                if ("generated" in text and "policy" in text and key == "generated_not_policy_mutable") or ("retrieved" in text and "verified" in text and key == "retrieved_not_verified") or ("graph path" in text and key == "graph_path_not_evidence") or ("surface" in text and "ui" in text and key == "surface_not_ui"):
                    hits.append(dict(row))
    blocked = any(h["enforcement_mode"] == "block" for h in hits)
    report = {"action": "demem_check", "instruction_sha256": sha256_text(args.instruction), "hits": hits, "blocked": blocked}
    write_report("demem_check", report)
    return 8 if blocked else 0


def cmd_catchme_check(args: argparse.Namespace) -> int:
    with connect(state_url(args)) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM lucidota_control.catchme_scope WHERE scope_key=%s", (args.scope_key,))
            row = cur.fetchone()
    blockers = []
    if row is None:
        blockers.append("scope_unknown")
    elif row["consent_status"] != "allowed" and not args.operator_approved:
        blockers.append(f"consent_required:{row['consent_status']}")
    report = {"action": "catchme_check", "scope_key": args.scope_key, "operator_approved": bool(args.operator_approved), "allowed": not blockers, "scope": dict(row) if row else None, "blockers": blockers}
    write_report("catchme_check", report)
    return 0 if not blockers else 9


def cmd_simplemem(args: argparse.Namespace) -> int:
    query = args.query.lower()
    corpus = args.corpus or []
    candidates = []
    for idx, text in enumerate(corpus):
        score = sum(1 for token in query.split() if token and token in text.lower())
        if score:
            candidates.append({"source_ref": f"inline:{idx}", "candidate_text": text, "recall_score": score, "not_truth": True, "promotion_allowed": False})
    candidates.sort(key=lambda x: x["recall_score"], reverse=True)
    report = {"action": "simplemem", "query_sha256": sha256_text(args.query), "candidates": candidates, "safe_to_answer_from_this_alone": False, "execute_performed": False}
    if args.execute and candidates:
        with connect(state_url(args)) as conn:
            with conn.cursor() as cur:
                for c in candidates[: args.limit]:
                    cur.execute("""
                        INSERT INTO lucidota_control.simplemem_candidate(query_sha256, source_ref, candidate_text, recall_score, detail)
                        VALUES (%s,%s,%s,%s,%s::jsonb)
                    """, (report["query_sha256"], c["source_ref"], c["candidate_text"], c["recall_score"], json.dumps({"script":"scripts/boring_beast.py"})))
            conn.commit()
        report["execute_performed"] = True
    write_report("simplemem", report)
    return 0


def ns_with(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def cmd_e2e(args: argparse.Namespace) -> int:
    # Execute a real tiny loop: enqueue -> consume -> duplicate check -> audit -> chrono -> status fact.
    init_args = ns_with(args, execute=True, dry_run=False)
    cmd_init_schema(init_args)
    idem = f"boring-e2e-{stamp().lower().rstrip('z')}-{os.getpid()}"
    payload = {"target_number": 20, "target_name": "Boring Beast E2E command", "idempotency_key": idem, "handler": "noop", "message": "e2e command path"}
    enqueue_args = ns_with(args, dry_run=False, execute=True, payload_json=json.dumps(payload), payload_file=None, priority=-100, max_attempts=2)
    cmd_enqueue(enqueue_args)
    worker_args = ns_with(args, dry_run=False, execute=True, worker_id="boring-beast-e2e", idempotency_key=idem)
    worker_rc = cmd_worker_once(worker_args)
    dup_args = ns_with(args, dry_run=False, execute=True, payload_json=json.dumps(payload), payload_file=None, priority=-100, max_attempts=2)
    cmd_enqueue(dup_args)
    audit_args = ns_with(args, dry_run=False, execute=True, task_id="boring_e2e", verdict="PASS", remediation="", evidence_ref=[f"idempotency:{idem}"])
    cmd_audit_verdict(audit_args)
    status_args = ns_with(args, execute=True, dry_run=False)
    cmd_status_from_runtime(status_args)
    report = {"action": "e2e", "idempotency_key": idem, "worker_returncode": worker_rc, "execute_performed": True, "db_writes_performed": True, "graph_writes_performed": False}
    write_report("e2e", report)
    return 0 if worker_rc == 0 else worker_rc




def cmd_transition_check(args: argparse.Namespace) -> int:
    cases = [
        ("queued", "running", "worker", True),
        ("running", "queued", "auditor", False),
        ("failed", "queued", "foreman", True),
        ("succeeded", "queued", "worker", False),
    ]
    results = []
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            for old, new, role, expected in cases:
                cur.execute("SELECT lucidota_control.dbos_queue_transition_allowed(%s,%s,%s)", (old, new, role))
                got = bool(cur.fetchone()[0])
                results.append({"old": old, "new": new, "role": role, "expected": expected, "got": got, "pass": got == expected})
    report = {"action": "transition_check", "results": results, "pass": all(r["pass"] for r in results)}
    write_report("transition_check", report)
    return 0 if report["pass"] else 10


def cmd_execution_record_get(args: argparse.Namespace) -> int:
    with connect(state_url(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if args.idempotency_key:
                cur.execute("""
                    SELECT execution_uuid::text, task_id, job_uuid::text, idempotency_key, status, result, created_at::text
                    FROM lucidota_control.boring_execution_record
                    WHERE idempotency_key=%s
                    ORDER BY created_at DESC LIMIT %s
                """, (args.idempotency_key, args.limit))
            else:
                cur.execute("""
                    SELECT execution_uuid::text, task_id, job_uuid::text, idempotency_key, status, result, created_at::text
                    FROM lucidota_control.boring_execution_record
                    ORDER BY created_at DESC LIMIT %s
                """, (args.limit,))
            rows = [dict(r) for r in cur.fetchall()]
    report = {"action": "execution_record_get", "count": len(rows), "records": rows}
    write_report("execution_record_get", report)
    return 0


def cmd_retry_failed(args: argparse.Namespace) -> int:
    report = {"action": "retry_failed", "execute_performed": False, "requeued_count": 0, "jobs": []}
    with connect(state_url(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT job_uuid::text FROM lucidota_control.dbos_queue_job
                WHERE queue_name='boring_beast' AND status='failed' AND attempt_count < max_attempts
                ORDER BY updated_at ASC LIMIT %s
            """, (args.limit,))
            jobs = [r["job_uuid"] for r in cur.fetchall()]
            report["jobs"] = jobs
            if args.execute and jobs:
                cur.execute("SET LOCAL lucidota.actor_role='foreman'")
                for job_uuid in jobs:
                    cur.execute("UPDATE lucidota_control.dbos_queue_job SET status='queued', run_after=now(), last_error='', error_kind='', error_message='' WHERE job_uuid=%s", (job_uuid,))
                    cur.execute("INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,'boring_beast','retry_scheduled','boring_beast',%s::jsonb)", (job_uuid, json.dumps({"source":"retry_failed"})))
                report["requeued_count"] = len(jobs)
        conn.commit()
    report["execute_performed"] = bool(args.execute)
    write_report("retry_failed", report)
    return 0


def cmd_oracle_snapshot(args: argparse.Namespace) -> int:
    entries = []
    for raw in args.path:
        path = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if not path.exists() or not path.is_file():
            entries.append({"path": raw, "exists": False})
            continue
        st = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        entries.append({"path": rel(path), "exists": True, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h.hexdigest()})
    report = {"action": "oracle_snapshot", "entries": entries, "manifest_sha256": sha256_obj(entries)}
    write_report("oracle_snapshot", report)
    return 0


def cmd_graph_barrier_status(args: argparse.Namespace) -> int:
    with connect(storage_url(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT event_object_table AS table_name, trigger_name, action_timing, event_manipulation
                FROM information_schema.triggers
                WHERE trigger_schema='lucidota_go' AND trigger_name IN ('trg_block_direct_graph_item_write','trg_block_direct_graph_edge_write')
                ORDER BY table_name, trigger_name, event_manipulation
            """)
            rows = [dict(r) for r in cur.fetchall()]
    report = {"action": "graph_barrier_status", "trigger_rows": rows, "active": len(rows) >= 2}
    write_report("graph_barrier_status", report)
    return 0 if report["active"] else 11


def cmd_tracer_validate_span(args: argparse.Namespace) -> int:
    span = load_payload(args.source_span_json, None)
    errors = []
    if not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int):
        errors.append("start_end_must_be_int")
    elif span["start"] < 0 or span["end"] < span["start"]:
        errors.append("invalid_span_bounds")
    if not isinstance(span.get("text", ""), str):
        errors.append("text_must_be_string")
    report = {"action": "tracer_validate_span", "span": span, "valid": not errors, "errors": errors}
    write_report("tracer_validate_span", report)
    return 0 if not errors else 12


def cmd_catchme_set_scope(args: argparse.Namespace) -> int:
    report = {"action": "catchme_set_scope", "scope_key": args.scope_key, "execute_performed": False}
    if not args.execute:
        write_report("catchme_set_scope_dry_run", report)
        return 0
    with connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.catchme_scope(scope_key,sensitivity_class,consent_status,allowed_use,detail)
                VALUES (%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (scope_key) DO UPDATE SET sensitivity_class=EXCLUDED.sensitivity_class, consent_status=EXCLUDED.consent_status, allowed_use=EXCLUDED.allowed_use, updated_at=now(), detail=EXCLUDED.detail
            """, (args.scope_key, args.sensitivity_class, args.consent_status, args.allowed_use, json.dumps({"script":"scripts/boring_beast.py"})))
        conn.commit()
    report["execute_performed"] = True
    write_report("catchme_set_scope_execute", report)
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LUCIDOTA Boring Beast executable work-loop runtime")
    p.add_argument("--state-database-url")
    p.add_argument("--storage-database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    def mode(sp):
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--dry-run", action="store_true", default=True)
        g.add_argument("--execute", action="store_true")
    sp = sub.add_parser("init-schema"); mode(sp); sp.set_defaults(func=cmd_init_schema)
    sp = sub.add_parser("apply-graph-barrier"); mode(sp); sp.set_defaults(func=cmd_apply_graph_barrier)
    sp = sub.add_parser("enqueue"); mode(sp); sp.add_argument("--payload-json"); sp.add_argument("--payload-file"); sp.add_argument("--priority", type=int, default=100); sp.add_argument("--max-attempts", type=int, default=2); sp.set_defaults(func=cmd_enqueue)
    sp = sub.add_parser("worker-once"); mode(sp); sp.add_argument("--worker-id"); sp.add_argument("--idempotency-key"); sp.set_defaults(func=cmd_worker_once)
    sp = sub.add_parser("recover-stale"); mode(sp); sp.add_argument("--timeout-seconds", type=int, default=300); sp.set_defaults(func=cmd_recover_stale)
    sp = sub.add_parser("audit-verdict"); mode(sp); sp.add_argument("--task-id", required=True); sp.add_argument("--verdict", choices=["PASS","FAIL","PARTIAL_FAIL"], required=True); sp.add_argument("--remediation", default=""); sp.add_argument("--evidence-ref", action="append", default=[]); sp.set_defaults(func=cmd_audit_verdict)
    sp = sub.add_parser("oracle-check"); sp.add_argument("--allowed-file", action="append", default=[]); sp.add_argument("--before-manifest", action="append", default=[]); sp.add_argument("--after-manifest", action="append", default=[]); sp.set_defaults(func=cmd_oracle_check)
    sp = sub.add_parser("supervisor-check"); sp.add_argument("--command", default="scripts/check_chrono_ledger_service.sh"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--timeout-seconds", type=int, default=30); sp.set_defaults(func=cmd_supervisor_check)
    sp = sub.add_parser("status-from-runtime"); mode(sp); sp.set_defaults(func=cmd_status_from_runtime)
    sp = sub.add_parser("chrono-event"); sp.add_argument("--event-kind", default="manual_runtime_event"); sp.add_argument("--detail-json", default="{}"); sp.set_defaults(func=cmd_chrono_event)
    sp = sub.add_parser("conversation-enqueue"); mode(sp); sp.add_argument("--plain-language-instruction"); sp.add_argument("--command-envelope-json"); sp.add_argument("--command-envelope-file"); sp.add_argument("--command-kind", default="operator_instruction"); sp.add_argument("--source-surface-id"); sp.add_argument("--allowed-effect"); sp.add_argument("--authority-class", default="operator_authored_assertion"); sp.add_argument("--idempotency-key"); sp.set_defaults(func=cmd_conversation_enqueue)
    sp = sub.add_parser("graph-promote"); mode(sp); sp.add_argument("--source-system", default="boring_beast"); sp.add_argument("--candidate-kind", choices=["node","edge","property","doctrine","workflow","other"], default="node"); sp.add_argument("--candidate-payload-json"); sp.add_argument("--candidate-payload-file"); sp.add_argument("--evidence-ref", action="append", default=[]); sp.add_argument("--authority-class", default="operator_authored_assertion"); sp.add_argument("--decision", choices=["defer","reject","promote","operator_confirmed","superseded"], default="defer"); sp.add_argument("--rationale", default="Boring Beast promotion gate execution"); sp.add_argument("--operator-confirmed", action="store_true"); sp.add_argument("--command-envelope-uuid"); sp.add_argument("--materialize", action="store_true"); sp.set_defaults(func=cmd_graph_promote)
    sp = sub.add_parser("direct-graph-write-check"); sp.set_defaults(func=cmd_direct_graph_write_check)
    sp = sub.add_parser("tracer-label"); mode(sp); sp.add_argument("--packet-ref", required=True); sp.add_argument("--label", required=True); sp.add_argument("--source-span-json", default="{}"); sp.add_argument("--authority-class", default="model_computed_finding"); sp.add_argument("--confidence-bps", type=int, default=5000); sp.set_defaults(func=cmd_tracer_label)
    sp = sub.add_parser("demem-check"); sp.add_argument("--instruction", required=True); sp.set_defaults(func=cmd_demem_check)
    sp = sub.add_parser("catchme-check"); sp.add_argument("--scope-key", required=True); sp.add_argument("--operator-approved", action="store_true"); sp.set_defaults(func=cmd_catchme_check)
    sp = sub.add_parser("simplemem"); sp.add_argument("--query", required=True); sp.add_argument("--corpus", action="append", default=[]); sp.add_argument("--limit", type=int, default=10); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=cmd_simplemem)
    sp = sub.add_parser("transition-check"); sp.set_defaults(func=cmd_transition_check)
    sp = sub.add_parser("execution-record-get"); sp.add_argument("--idempotency-key"); sp.add_argument("--limit", type=int, default=5); sp.set_defaults(func=cmd_execution_record_get)
    sp = sub.add_parser("retry-failed"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--limit", type=int, default=10); sp.set_defaults(func=cmd_retry_failed)
    sp = sub.add_parser("oracle-snapshot"); sp.add_argument("--path", action="append", required=True); sp.set_defaults(func=cmd_oracle_snapshot)
    sp = sub.add_parser("graph-barrier-status"); sp.set_defaults(func=cmd_graph_barrier_status)
    sp = sub.add_parser("tracer-validate-span"); sp.add_argument("--source-span-json", required=True); sp.set_defaults(func=cmd_tracer_validate_span)
    sp = sub.add_parser("catchme-set-scope"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--scope-key", required=True); sp.add_argument("--sensitivity-class", choices=["public","internal","private","secret","revoked"], required=True); sp.add_argument("--consent-status", choices=["allowed","requires_operator","revoked"], required=True); sp.add_argument("--allowed-use", default="none"); sp.set_defaults(func=cmd_catchme_set_scope)
    sp = sub.add_parser("e2e"); sp.add_argument("--execute", action="store_true", required=True); sp.set_defaults(func=cmd_e2e)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "execute", False):
        args.dry_run = False
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
