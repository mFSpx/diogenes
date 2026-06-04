#!/usr/bin/env python3
"""ABSURD graph-promotion worker.

Consumes only registry-approved graph_promotion jobs. Packet staging is routed
through scripts/graph_promotion_gate.py so this worker cannot bypass evidence,
authority, or preflight checks. Canonical graph materialization is not performed
here.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
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

from absurd_worker_contracts import WorkerContractDecision
from absurd_worker_contracts import gate_worker_payload_hygiene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "absurd"
SCHEMAS = [
    ROOT / "06_SCHEMA/035_absurd_queue_spine.sql",
    ROOT / "06_SCHEMA/039_absurd_real_work_loop.sql",
    ROOT / "06_SCHEMA/043_absurd_remaining_worker_contracts.sql",
    ROOT / "06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql",
]
QUEUE = "graph_promotion"
WORKFLOW = "absurd-graph-promotion"
WORKER_KEY = "graph_promotion_worker"
WAKEUP_CHANNEL = "lucidota_queue_wakeup"
MAX_PAYLOAD_JSON_BYTES = 65536
DEFAULT_PAYLOAD = {
    "candidate_payload": {"term": "CLAIM", "label": "graph promotion wrapper health", "evidence_note": "wrapper health"},
    "evidence_refs": ["absurd_graph_promotion_worker"],
    "authority_class": "operator_authored_assertion",
    "decision": "defer",
}


@dataclass(frozen=True)
class ClaimedJob:
    job_uuid: str
    payload: dict[str, Any]
    idempotency_key: str
    job_kind: str
    attempts: int
    max_attempts: int
    contract: WorkerContractDecision


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def state_url(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def storage_url(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"absurd_graph_promotion_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def load_payload(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return DEFAULT_PAYLOAD
    size = len(raw.encode("utf-8"))
    if size > MAX_PAYLOAD_JSON_BYTES:
        raise ValueError(f"payload_json_too_large:{size}>{MAX_PAYLOAD_JSON_BYTES}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("payload_json_must_be_object")
    return payload


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_url(args)) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
            conn.commit()
    write("init_schema_execute" if args.execute else "init_schema_dry_run", {"action": "init_schema", "execute_performed": bool(args.execute), "schemas": [rel(s) for s in SCHEMAS]})
    return 0


def enqueue(args: argparse.Namespace) -> int:
    try:
        payload = load_payload(args.payload_json)
    except Exception as exc:
        write("enqueue_blocked", {"action": "enqueue", "execute_performed": False, "inserted_new": False, "error": str(exc), "status": "BLOCKED"})
        return 2
    idem = args.idempotency_key or sha(payload)
    result = {"action": "enqueue", "queue": QUEUE, "job_kind": "graph_promotion_packet_defer", "idempotency_key": idem, "execute_performed": False, "inserted_new": False}
    if not args.execute:
        write("enqueue_dry_run", result)
        return 0
    with psycopg.connect(state_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_job(queue_name,workflow_name,job_kind,idempotency_key,payload)
                VALUES (%s,%s,'graph_promotion_packet_defer',%s,%s::jsonb)
                ON CONFLICT(queue_name,idempotency_key) DO UPDATE SET updated_at=lucidota_control.absurd_queue_job.updated_at
                RETURNING job_uuid::text,(xmax=0)
                """,
                (QUEUE, WORKFLOW, idem, json.dumps(payload)),
            )
            job, inserted = cur.fetchone()
            cur.execute("SELECT pg_notify(%s,%s)", (WAKEUP_CHANNEL, json.dumps({"queue": QUEUE, "job_uuid": job})))
            result.update({"job_uuid": job, "inserted_new": bool(inserted)})
        conn.commit()
    result["execute_performed"] = True
    write("enqueue_execute", result)
    print(f"JOB_UUID={job}")
    return 0


def run_gate(args: argparse.Namespace, job_uuid: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    decision = str(payload.get("decision") or "defer")
    if decision != "defer":
        return False, {"error": "graph_promotion_worker_only_defers", "decision": decision}
    evidence_refs = payload.get("evidence_refs") or []
    if not isinstance(evidence_refs, list) or not evidence_refs:
        return False, {"error": "evidence_refs_required"}
    candidate_payload = payload.get("candidate_payload") or {}
    if not isinstance(candidate_payload, dict):
        return False, {"error": "candidate_payload_must_be_object"}
    cmd = [
        sys.executable,
        "scripts/graph_promotion_gate.py",
        "--database-url",
        storage_url(args),
        "gate",
        "--execute",
        "--source-system",
        "absurd_graph_promotion_worker",
        "--candidate-kind",
        str(payload.get("candidate_kind") or "node"),
        "--candidate-payload-json",
        json.dumps(candidate_payload, sort_keys=True),
        "--authority-class",
        str(payload.get("authority_class") or "operator_authored_assertion"),
        "--decision",
        "defer",
        "--rationale",
        f"ABSURD graph promotion worker deferred packet for job {job_uuid}",
    ]
    for ref in evidence_refs:
        cmd.extend(["--evidence-ref", str(ref)])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=int(payload.get("timeout_seconds", 120)))
    report_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            report_path = line.split("=", 1)[1]
    result = {"returncode": proc.returncode, "report_path": report_path, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:], "routed_through": "scripts/graph_promotion_gate.py"}
    if report_path and (ROOT / report_path).exists():
        try:
            result["gate_report"] = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
        except Exception as exc:
            result["gate_report_error"] = str(exc)
    return proc.returncode == 0, result


def _split_job_kinds(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    text = str(raw or "")
    return tuple(x.strip() for x in text.replace(",", "|").split("|") if x.strip())


def _script_exists(script_path: str) -> bool:
    p = Path(script_path)
    if not p.is_absolute():
        p = ROOT / p
    return p.exists() and p.is_file()


async def validate_worker_contract_async(cur: Any, *, queue_name: str, job_kind: str, worker_key: str | None = None) -> WorkerContractDecision:
    await cur.execute("SELECT to_regclass('lucidota_control.absurd_worker_contract')")
    row = await cur.fetchone()
    exists = row[0] if not isinstance(row, dict) else row.get("to_regclass")
    if exists is None:
        return WorkerContractDecision(False, "worker_contract_table_missing", "lucidota_control.absurd_worker_contract is missing", queue_name=queue_name, job_kind=job_kind)

    params: list[Any] = [queue_name]
    worker_clause = ""
    if worker_key:
        worker_clause = " AND worker_key=%s"
        params.append(worker_key)
    await cur.execute(
        f"""
        SELECT worker_key, queue_name, script_path, input_contract, canonical_graph_write_allowed, status
        FROM lucidota_control.absurd_worker_contract
        WHERE queue_name=%s{worker_clause}
          AND status IN ('implemented','verified')
        ORDER BY updated_at DESC, created_at DESC
        """,
        params,
    )
    rows = await cur.fetchall()
    if not rows:
        return WorkerContractDecision(False, "worker_contract_not_found", f"no active contract for queue {queue_name}", queue_name=queue_name, job_kind=job_kind)

    seen: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            wk, qn, sp, contract = row["worker_key"], row["queue_name"], row["script_path"], row["input_contract"]
        else:
            wk, qn, sp, contract = row[0], row[1], row[2], row[3]
        if isinstance(contract, str):
            try:
                contract = json.loads(contract)
            except Exception:
                contract = {}
        allowed = _split_job_kinds((contract or {}).get("job_kind"))
        seen.extend(allowed)
        if job_kind in allowed:
            if not _script_exists(str(sp)):
                return WorkerContractDecision(False, "worker_contract_script_missing", f"contract script missing: {sp}", worker_key=str(wk), queue_name=str(qn), job_kind=job_kind, script_path=str(sp), allowed_job_kinds=allowed)
            return WorkerContractDecision(True, worker_key=str(wk), queue_name=str(qn), job_kind=job_kind, script_path=str(sp), allowed_job_kinds=allowed)

    return WorkerContractDecision(False, "job_kind_not_in_worker_contract", f"{job_kind} not allowed for {queue_name}; allowed={sorted(set(seen))}", queue_name=queue_name, job_kind=job_kind, allowed_job_kinds=tuple(sorted(set(seen))))


async def worker_dry_run(args: argparse.Namespace) -> int:
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {"action": "worker_once", "worker_id": worker_id, "execute_performed": False, "job_processed": False, "canonical_graph_writes_performed": False}
    async with await psycopg.AsyncConnection.connect(state_url(args)) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT job_uuid::text,job_kind FROM lucidota_control.absurd_queue_job WHERE queue_name=%s AND status='queued' ORDER BY created_at LIMIT 1", (QUEUE,))
            row = await cur.fetchone()
            result["would_process"] = {"job_uuid": row[0], "job_kind": row[1]} if row else None
            if row:
                result["worker_contract"] = (await validate_worker_contract_async(cur, queue_name=QUEUE, job_kind=str(row[1]), worker_key=WORKER_KEY)).as_result()
        await conn.commit()
    write("worker_dry_run", result)
    return 0


async def claim_durable_job(args: argparse.Namespace, worker_id: str) -> ClaimedJob | None:
    async with await psycopg.AsyncConnection.connect(state_url(args)) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET LOCAL lucidota.actor_role='worker'")
            await cur.execute(
                """
                SELECT job_uuid::text,payload,idempotency_key,job_kind,attempt_count,max_attempts
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s AND status='queued'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (QUEUE,),
            )
            row = await cur.fetchone()
            if not row:
                await conn.rollback()
                return None
            job_uuid, payload, idem, job_kind, attempts, max_attempts = row
            payload = dict(payload or {})
            contract = await validate_worker_contract_async(cur, queue_name=QUEUE, job_kind=str(job_kind), worker_key=WORKER_KEY)
            await cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='running',locked_by=%s,locked_at=now(),attempt_count=attempt_count+1 WHERE job_uuid=%s", (worker_id, job_uuid))
        await conn.commit()
    return ClaimedJob(str(job_uuid), payload, str(idem), str(job_kind), int(attempts), int(max_attempts), contract)


async def finish_durable_job(args: argparse.Namespace, *, status: str, gate_result: dict[str, Any], job_uuid: str, ok: bool) -> None:
    async with await psycopg.AsyncConnection.connect(state_url(args)) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET LOCAL lucidota.actor_role='worker'")
            await cur.execute("UPDATE lucidota_control.absurd_queue_job SET status=%s,result=%s::jsonb,completed_at=CASE WHEN %s='succeeded' THEN now() ELSE completed_at END,last_error=%s WHERE job_uuid=%s", (status, json.dumps(gate_result, default=str), status, "" if ok else json.dumps(gate_result, default=str), job_uuid))
            await cur.execute("INSERT INTO lucidota_control.workflow_event(workflow_id,run_id,phase,status,source,detail) VALUES (%s,%s,'graph_promotion',%s,'absurd_graph_promotion_worker',%s::jsonb)", (WORKFLOW, job_uuid, status, json.dumps(gate_result, default=str)))
        await conn.commit()


async def attempt_durable_claim_result(args: argparse.Namespace, worker_id: str | None = None, *, emit_no_job_report: bool = False) -> tuple[bool, bool]:
    worker_id = worker_id or args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {"action": "worker_once", "worker_id": worker_id, "execute_performed": False, "job_processed": False, "canonical_graph_writes_performed": False}
    claimed = await claim_durable_job(args, worker_id)
    if claimed is None:
        if emit_no_job_report:
            result["no_job_available"] = True
            write("worker_execute", result)
        return False, True

    job_uuid = claimed.job_uuid
    payload = claimed.payload
    job_kind = claimed.job_kind
    attempts = claimed.attempts
    max_attempts = claimed.max_attempts
    contract = claimed.contract

    if not contract.ok:
        ok = False
        gate_result = {"error": "worker_contract_rejected", "worker_contract": contract.as_result()}
    elif job_kind == "graph_promotion_health_check":
        ok = True
        gate_result = {"ok": True, "message": "graph promotion worker health check"}
    else:
        ok, gate_result = await asyncio.to_thread(run_gate, args, job_uuid, payload)
    if ok:
        gate_payload = {"result": gate_result, "outcome": "succeeded"}
        payload_ok, hygiene = gate_worker_payload_hygiene(
            gate_payload,
            queue_name=QUEUE,
            worker_key=WORKER_KEY,
            job_kind=job_kind,
            required_keys=(),
            min_score=0,
        )
        if not payload_ok:
            ok = False
            gate_result = {"result": gate_payload, "hygiene": hygiene, "outcome": "failed"}

    status = "succeeded" if ok else ("dead_lettered" if int(attempts) + 1 >= int(max_attempts) else "failed")
    await finish_durable_job(args, status=status, gate_result=gate_result, job_uuid=job_uuid, ok=ok)
    result.update({"execute_performed": True, "job_processed": True, "job_uuid": job_uuid, "job_kind": job_kind, "status": status, "result": gate_result, "worker_contract": contract.as_result(), "canonical_graph_writes_performed": False})
    write("worker_execute", result)
    print(f"JOB_UUID={job_uuid}")
    return True, ok


async def attempt_durable_claim(args: argparse.Namespace, worker_id: str | None = None, *, emit_no_job_report: bool = False) -> bool:
    claimed, ok = await attempt_durable_claim_result(args, worker_id=worker_id, emit_no_job_report=emit_no_job_report)
    setattr(args, "_last_claim_ok", ok)
    return claimed


async def worker_once_async(args: argparse.Namespace) -> int:
    if not args.execute:
        return await worker_dry_run(args)
    claimed, ok = await attempt_durable_claim_result(args, emit_no_job_report=True)
    if not claimed:
        return 0
    return 0 if ok else 3


def worker_once(args: argparse.Namespace) -> int:
    return asyncio.run(worker_once_async(args))


async def drain_durable_queue(args: argparse.Namespace, worker_id: str) -> None:
    while await attempt_durable_claim(args, worker_id=worker_id):
        pass


async def worker_listen_async(args: argparse.Namespace) -> int:
    if not args.execute:
        return await worker_dry_run(args)
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    async with await psycopg.AsyncConnection.connect(state_url(args), autocommit=True) as listen_conn:
        async with listen_conn.cursor() as cur:
            await cur.execute(f"LISTEN {WAKEUP_CHANNEL}")
        await drain_durable_queue(args, worker_id)
        async for _notify in listen_conn.notifies():
            await drain_durable_queue(args, worker_id)
    return 0


def worker_listen(args: argparse.Namespace) -> int:
    return asyncio.run(worker_listen_async(args))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-database-url")
    parser.add_argument("--storage-database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in [("init-schema", init_schema), ("enqueue", enqueue), ("worker-once", worker_once), ("worker", worker_listen), ("worker-listen", worker_listen)]:
        sp = sub.add_parser(name)
        sp.add_argument("--execute", action="store_true")
        sp.set_defaults(func=func)
        if name == "enqueue":
            sp.add_argument("--payload-json")
            sp.add_argument("--idempotency-key")
        if name in {"worker-once", "worker", "worker-listen"}:
            sp.add_argument("--worker-id")
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
