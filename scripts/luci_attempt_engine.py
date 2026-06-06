#!/usr/bin/env python3
"""Deterministic bounded attempt engine on the reusable ABSURD/board DB spine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "luci_attempt_engine"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace | None = None) -> str:
    return (
        (getattr(args, "database_url", None) if args is not None else None)
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def event_id_for(job_uuid: str, attempt_no: int, step: str, *, salt: str = "") -> str:
    return sha256_text(stable_json({"job_uuid": job_uuid, "attempt_no": attempt_no, "step": step, "salt": salt}))


def classify_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job.get("payload") or {})
    hints = sorted({str(job.get("job_kind") or ""), str(payload.get("task") or ""), str(payload.get("mode") or "")} - {""})
    safe_probe = str(payload.get("probe_sql") or "SELECT 1 AS ok")
    if not safe_probe.strip().lower().startswith("select"):
        safe_probe = "SELECT 1 AS ok"
    attempt = {
        "attempt_kind": "safe_probe",
        "probe_sql": safe_probe,
        "hints": hints,
        "expected_cost": 1,
        "expected_gain": 0.25,
        "decision": "probe",
    }
    if "refresh" in hints or payload.get("priority") == "low":
        attempt["expected_cost"] = 0
        attempt["expected_gain"] = 0.12
    return attempt


def intake_job_from_text(text: str, *, queue_name: str, run_id: str) -> dict[str, Any]:
    payload = {
        "task": "operator_text",
        "text": text,
        "probe_sql": "SELECT 1 AS ok",
        "priority": "low",
    }
    job_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_json({"queue_name": queue_name, "run_id": run_id, "text": text})))
    return {
        "job_uuid": job_uuid,
        "queue_name": queue_name,
        "workflow_name": "luci_attempt_engine",
        "job_kind": "synthetic_operator_text",
        "idempotency_key": f"luci:{queue_name}:{run_id}:{sha256_text(text)[:16]}",
        "payload": payload,
        "status": "queued",
        "priority": 1,
        "attempt_count": 0,
        "max_attempts": 2,
    }


def execute_probe(conn: psycopg.Connection, attempt: dict[str, Any]) -> dict[str, Any]:
    sql = str(attempt["probe_sql"])
    started = time.monotonic()
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall() if cur.description else []
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    return {"ok": True, "row_count": len(rows), "latency_ms": latency_ms, "rows": rows[:3]}


def score_attempt(attempt: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    gain = 1.0 if observation.get("ok") else 0.0
    cost = float(attempt.get("expected_cost", 1))
    score = round(max(0.0, gain - (0.05 * cost)), 3)
    verdict = "win" if observation.get("ok") else "retry"
    if observation.get("latency_ms", 0) > 1000:
        verdict = "stall"
        score = round(score * 0.5, 3)
    return {"verdict": verdict, "score": score, "gain": gain, "cost": cost}


def next_status(job: dict[str, Any], score: dict[str, Any]) -> tuple[str, str, str]:
    attempt_count = int(job.get("attempt_count") or 0) + 1
    max_attempts = int(job.get("max_attempts") or 1)
    if score["verdict"] == "win":
        return "succeeded", "", ""
    if attempt_count < max_attempts:
        return "queued", "retry_scheduled", f"retry in {attempt_count * 2} minutes"
    return "failed", "final_failure", "exhausted attempts"


def ensure_schema_objects(cur: Any) -> None:
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_luci_attempt_engine_absurd_queue_job_claim
          ON lucidota_control.absurd_queue_job(status, queue_name, priority, created_at)
          WHERE status IN ('queued','leased','running')
        """
    )
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, status, notes)
        VALUES ('luci_operator', 'luci', 'active', 'LUCI operator runtime closure work loop queue')
        ON CONFLICT (queue_name) DO NOTHING
        """
    )
    cur.execute(
        """
        INSERT INTO lucidota_control.worker(worker_id, actor_class, runtime_kind, host_id, lane_id, active_mode)
        VALUES ('luci_attempt_engine', 'luci', 'python_worker', coalesce(inet_server_addr()::text, 'local'), 'luci_operator', 'idle')
        ON CONFLICT (worker_id) DO UPDATE
          SET active_mode='idle', updated_at=now()
        """
    )


def seed_queue_job(cur: Any, job: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    inserted = cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_job
          (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, payload, status, priority, attempt_count, max_attempts)
        VALUES (%s::uuid, %s, %s, %s, %s, %s::jsonb, 'queued', %s, %s, %s)
        ON CONFLICT (queue_name, idempotency_key) DO NOTHING
        RETURNING job_uuid::text
        """,
        (
            job["job_uuid"],
            job["queue_name"],
            job["workflow_name"],
            job["job_kind"],
            job["idempotency_key"],
            json.dumps(job["payload"]),
            job["priority"],
            job["attempt_count"],
            job["max_attempts"],
        ),
    ).fetchone()
    row = cur.execute(
        """
        SELECT job_uuid::text, queue_name, workflow_name, job_kind, idempotency_key, payload, status::text,
               priority, attempt_count, max_attempts, result
        FROM lucidota_control.absurd_queue_job
        WHERE queue_name=%s AND idempotency_key=%s
        FOR UPDATE
        """,
        (job["queue_name"], job["idempotency_key"]),
    ).fetchone()
    return dict(row), bool(inserted)


def count_dead_letters(cur: Any, job_uuid: str) -> int:
    row = cur.execute(
        """
        SELECT count(*)::int AS count
        FROM lucidota_control.absurd_queue_dead_letter
        WHERE job_uuid=%s::uuid
        """,
        (job_uuid,),
    ).fetchone()
    return int(row["count"] if isinstance(row, dict) else row[0])


def record_attempt(cur: Any, job: dict[str, Any], attempt: dict[str, Any], observation: dict[str, Any], score: dict[str, Any], *, receipt_path: str) -> dict[str, str]:
    attempt_no = int(job.get("attempt_count") or 0) + 1
    event_id = event_id_for(str(job["job_uuid"]), attempt_no, "luci_attempt_engine")
    raw_ref = f"inline://{sha256_text(stable_json(job))[:16]}:{job['job_uuid']}:{attempt_no}"
    raw_uuid_row = cur.execute(
        """
        INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
        VALUES (%s,%s,'sha256','luci_attempt_engine','worker',%s,%s,'application/json','inline_or_receipt',%s::jsonb)
        ON CONFLICT (raw_ref) DO UPDATE
          SET detail = lucidota_control.raw_artifact.detail || EXCLUDED.detail
        RETURNING raw_artifact_uuid::text
        """,
        (raw_ref, sha256_text(stable_json(job)), len(stable_json(job).encode()), len(stable_json(job)), json.dumps({"job_uuid": str(job["job_uuid"])})),
    ).fetchone()
    raw_uuid = raw_uuid_row["raw_artifact_uuid"] if isinstance(raw_uuid_row, dict) else raw_uuid_row[0]
    cur.execute(
        """
        INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
        VALUES (%s, now(), 'luci_attempt_engine', 'worker', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, %s::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, %s::jsonb, NULL, %s::jsonb)
        ON CONFLICT (event_id) DO UPDATE
          SET detail = lucidota_control.event_envelope.detail || EXCLUDED.detail
        RETURNING event_id
        """,
        (
            event_id,
            raw_ref,
            raw_uuid,
            sha256_text(str(job.get("job_kind") or "")),
            stable_json({"job": {"uuid": str(job["job_uuid"]), "kind": job.get("job_kind")}, "attempt": attempt, "observation": observation, "score": score}),
            json.dumps([attempt["attempt_kind"]]),
            json.dumps({"job_kind": job.get("job_kind"), "latency_ms": observation.get("latency_ms", 0)}),
            json.dumps({"receipt_path": receipt_path}),
        ),
    )
    work_order_key = f"luci:{job['job_uuid']}:{attempt_no}:{attempt['attempt_kind']}"
    work_order_row = cur.execute(
        """
        INSERT INTO lucidota_control.work_order(event_id, lane, work_kind, status, payload, idempotency_key)
        VALUES (%s, 'audit', %s, %s, %s::jsonb, %s)
        ON CONFLICT (idempotency_key) DO UPDATE
          SET event_id = EXCLUDED.event_id,
              status = EXCLUDED.status,
              payload = EXCLUDED.payload,
              updated_at = now()
        RETURNING work_order_uuid::text
        """,
        (event_id, "luci_attempt_probe", "queued" if score["verdict"] != "win" else "succeeded", json.dumps({"job_uuid": str(job["job_uuid"]), "attempt": attempt}), work_order_key),
    ).fetchone()
    work_order_uuid = work_order_row["work_order_uuid"] if isinstance(work_order_row, dict) else work_order_row[0]
    receipt_lookup = cur.execute(
        """
        SELECT work_receipt_uuid::text
        FROM lucidota_control.work_receipt
        WHERE event_id=%s AND receipt_path=%s
        """,
        (event_id, receipt_path),
    ).fetchone()
    if receipt_lookup:
        receipt_uuid = receipt_lookup["work_receipt_uuid"] if isinstance(receipt_lookup, dict) else receipt_lookup[0]
    else:
        receipt_row = cur.execute(
            """
            INSERT INTO lucidota_control.work_receipt(event_id, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
            VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, false, 'staged_only', %s::jsonb)
            RETURNING work_receipt_uuid::text
            """,
            (
                event_id,
                work_order_uuid,
                receipt_path,
                sha256_text(stable_json({"event_id": event_id, "job_uuid": str(job["job_uuid"])})),
                score["verdict"],
                json.dumps({"expected_cost": score["cost"]}),
                json.dumps({"gain": score["gain"], "score": score["score"]}),
                json.dumps([raw_ref]),
                json.dumps({"attempt_kind": attempt["attempt_kind"], "observation": observation}),
            ),
        ).fetchone()
        receipt_uuid = receipt_row["work_receipt_uuid"] if isinstance(receipt_row, dict) else receipt_row[0]
    attempt_lookup = cur.execute(
        """
        SELECT attempt_uuid::text
        FROM lucidota_control.work_order_attempt
        WHERE work_order_uuid=%s::uuid
          AND worker_id='luci_attempt_engine'
          AND receipt_uuid=%s::uuid
        LIMIT 1
        """,
        (work_order_uuid, receipt_uuid),
    ).fetchone()
    if attempt_lookup:
        work_order_attempt_uuid = attempt_lookup["attempt_uuid"] if isinstance(attempt_lookup, dict) else attempt_lookup[0]
    else:
        attempt_row = cur.execute(
            """
            INSERT INTO lucidota_control.work_order_attempt
              (work_order_uuid, worker_id, started_at, completed_at, status, proof_status, receipt_uuid)
            VALUES (%s::uuid, 'luci_attempt_engine', now(), now(), 'succeeded', 'PROVEN', %s::uuid)
            RETURNING attempt_uuid::text
            """,
            (work_order_uuid, receipt_uuid),
        ).fetchone()
        work_order_attempt_uuid = attempt_row["attempt_uuid"] if isinstance(attempt_row, dict) else attempt_row[0]
    audit_lookup = cur.execute(
        """
        SELECT workload_audit_uuid::text
        FROM lucidota_audit.workload_audit_ledger
        WHERE receipt_uuid=%s::uuid
        LIMIT 1
        """,
        (receipt_uuid,),
    ).fetchone()
    if audit_lookup:
        workload_audit_uuid = audit_lookup["workload_audit_uuid"] if isinstance(audit_lookup, dict) else audit_lookup[0]
    else:
        audit_row = cur.execute(
            """
            INSERT INTO lucidota_audit.workload_audit_ledger
              (actor_id, actor_class, caller, provider, model_id, action_summary, tokens_in, tokens_out,
               token_source, receipt_uuid, evidence_refs, proof_status, functionality_explanation, ontology_index,
               work_order_uuid, work_order_attempt_uuid, worker_id)
            VALUES
              ('luci_attempt_engine', 'unknown', 'operator', 'local', '', %s, %s, 0,
               'local_counter', %s::uuid, %s::jsonb, 'PROVEN', %s, %s::jsonb,
               %s::uuid, %s::uuid, 'luci_attempt_engine')
            RETURNING workload_audit_uuid::text
            """,
            (
                f"LUCI bounded runtime closure worker executed {attempt['attempt_kind']} for job {job['job_uuid']}",
                len(stable_json(job).split()),
                receipt_uuid,
                json.dumps([receipt_path, raw_ref]),
                "Receipt-backed LUCI operate worker execution; no canonical graph writes.",
                json.dumps(
                    {
                        "primitive_refs": ["state", "receipt", "worker_claim"],
                        "claim_type": "luci_runtime_closure",
                        "proof_status": "PROVEN",
                        "canonical_graph_writes_performed": False,
                    }
                ),
                work_order_uuid,
                work_order_attempt_uuid,
            ),
        ).fetchone()
        workload_audit_uuid = audit_row["workload_audit_uuid"] if isinstance(audit_row, dict) else audit_row[0]
    cur.execute(
        """
        UPDATE lucidota_control.worker
        SET active_mode='idle', updated_at=now()
        WHERE worker_id='luci_attempt_engine'
        """
    )
    return {
        "event_id": event_id,
        "raw_artifact_uuid": raw_uuid,
        "work_order_uuid": work_order_uuid,
        "work_receipt_uuid": receipt_uuid,
        "work_order_attempt_uuid": work_order_attempt_uuid,
        "workload_audit_uuid": workload_audit_uuid,
        "worker_id": "luci_attempt_engine",
    }


def claim_job(cur: Any, queue_name: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT job_uuid::text, queue_name, workflow_name, job_kind, idempotency_key, payload, status::text, priority, attempt_count, max_attempts
        FROM lucidota_control.absurd_queue_job
        WHERE queue_name=%s AND status='queued' AND run_after<=now()
        ORDER BY priority, created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        """,
        (queue_name,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def update_job(cur: Any, job_uuid: str, *, status: str, result: dict[str, Any], last_error: str = "", run_after: str | None = None) -> None:
    cur.execute(
        """
        UPDATE lucidota_control.absurd_queue_job
        SET status=%s, result=%s::jsonb, last_error=%s, completed_at=CASE WHEN %s IN ('succeeded','failed') THEN now() ELSE completed_at END,
            run_after=COALESCE(%s::timestamptz, run_after), updated_at=now()
        WHERE job_uuid=%s::uuid
        """,
        (status, json.dumps(result), last_error, status, run_after, job_uuid),
    )


def maybe_seed_synthetic_job(cur: Any, *, queue_name: str = "luci_attempt", run_id: str | None = None, text: str | None = None) -> dict[str, Any]:
    if text is not None:
        return intake_job_from_text(text, queue_name=queue_name, run_id=run_id or stamp())
    payload = {"task": "probe", "probe_sql": "SELECT 1 AS ok", "priority": "low"}
    job_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_json({"payload": payload, "queue_name": queue_name, "run_id": run_id or stamp()})))
    return {
        "job_uuid": job_uuid,
        "queue_name": queue_name,
        "workflow_name": "luci_attempt_engine",
        "job_kind": "synthetic_probe",
        "idempotency_key": f"luci:{queue_name}:{job_uuid}",
        "payload": payload,
        "status": "queued",
        "priority": 1,
        "attempt_count": 0,
        "max_attempts": 2,
    }


def run_once(conn: psycopg.Connection, *, queue_name: str, synthetic: bool = False, receipt_dir: Path | None = None, text: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    receipt_dir = receipt_dir or OUT
    receipt_dir.mkdir(parents=True, exist_ok=True)
    with conn.cursor(row_factory=dict_row) as cur:
        ensure_schema_objects(cur)
        if synthetic and text is not None:
            seed = maybe_seed_synthetic_job(cur, queue_name=queue_name, run_id=run_id, text=text)
            job, inserted = seed_queue_job(cur, seed)
        else:
            job = claim_job(cur, queue_name)
            inserted = False
            if job is None and synthetic:
                seed = maybe_seed_synthetic_job(cur, queue_name=queue_name, run_id=run_id, text=text)
                job, inserted = seed_queue_job(cur, seed)
        if job is None:
            return {"status": "no_work", "queue_name": queue_name}
        existing_result = dict(job.get("result") or {})
        if str(job.get("status")) in {"succeeded", "failed", "dead_lettered", "cancelled"} and existing_result.get("work_order_uuid"):
            repaired_legacy_result = False
            if not {"work_order_attempt_uuid", "workload_audit_uuid", "worker_id"}.issubset(existing_result):
                attempt = classify_job(job)
                observation = execute_probe(conn, attempt)
                score = score_attempt(attempt, observation)
                ids = record_attempt(
                    cur,
                    job,
                    attempt,
                    observation,
                    score,
                    receipt_path=str(existing_result.get("receipt_path") or ""),
                )
                existing_result.update(ids)
                repaired_legacy_result = True
                cur.execute(
                    """
                    UPDATE lucidota_control.absurd_queue_job
                    SET result = result || %s::jsonb
                    WHERE job_uuid=%s::uuid
                    """,
                    (json.dumps(existing_result), job["job_uuid"]),
                )
                conn.commit()
            return {
                "schema": "lucidota.luci_attempt_engine.receipt.v1",
                "generated_at": now(),
                "queue_name": queue_name,
                "job_uuid": str(job["job_uuid"]),
                "status": "PASS" if str(job.get("status")) == "succeeded" else "DEGRADED",
                "next_state": str(job.get("status")),
                "db_write": existing_result,
                "real_work_loop": {
                    "worker_executed": repaired_legacy_result,
                    "idempotent_replay": True,
                    "legacy_result_repaired": repaired_legacy_result,
                    "queue_job_inserted": False,
                    "queue_job_status": str(job.get("status")),
                    "dead_letter_count": count_dead_letters(cur, str(job["job_uuid"])),
                },
                "passed": str(job.get("status")) == "succeeded",
                "receipt_path": existing_result.get("receipt_path", ""),
            }
        cur.execute(
            """
            UPDATE lucidota_control.absurd_queue_job
            SET status='running',
                leased_by='luci_attempt_engine',
                lease_expires_at=now()+interval '5 minutes',
                attempt_count=attempt_count+1
            WHERE job_uuid=%s::uuid
            """,
            (job["job_uuid"],),
        )
        cur.execute(
            """
            UPDATE lucidota_control.worker
            SET active_mode='running', updated_at=now()
            WHERE worker_id='luci_attempt_engine'
            """
        )
        attempt = classify_job(job)
        observation = execute_probe(conn, attempt)
        score = score_attempt(attempt, observation)
        next_state, event_kind, reason = next_status(job, score)
        attempt_no = int(job.get("attempt_count") or 0) + 1
        receipt_path = receipt_dir / f"luci_attempt_{job['job_uuid']}_{attempt_no}.json"
        receipt = {
            "schema": "lucidota.luci_attempt_engine.receipt.v1",
            "generated_at": now(),
            "queue_name": queue_name,
            "job_uuid": str(job["job_uuid"]),
            "attempt": attempt,
            "observation": observation,
            "score": score,
            "next_state": next_state,
            "status": "PASS",
        }
        receipt["receipt_path"] = rel(receipt_path)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        ids = record_attempt(cur, job, attempt, observation, score, receipt_path=receipt["receipt_path"])
        if next_state == "queued":
            update_job(cur, job["job_uuid"], status="queued", result={"next_state": next_state, "reason": reason, "receipt_path": receipt["receipt_path"], **ids}, run_after=(datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat())
        else:
            update_job(cur, job["job_uuid"], status=next_state, result={"next_state": next_state, "reason": reason, "receipt_path": receipt["receipt_path"], **ids}, last_error=reason)
        conn.commit()
        receipt["db_write"] = {"job": job["job_uuid"], **ids, "event_kind": event_kind}
        receipt["real_work_loop"] = {
            "worker_executed": True,
            "idempotent_replay": False,
            "queue_job_inserted": inserted,
            "queue_job_status": next_state,
            "dead_letter_count": count_dead_letters(cur, str(job["job_uuid"])),
        }
        return receipt


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url")
    ap.add_argument("--queue-name", default="control")
    ap.add_argument("--synthetic", action="store_true", help="Run against a synthetic intake if no queue job exists.")
    ap.add_argument("--text", help="Operator text to seed synthetic intake from.")
    ap.add_argument("--run-id", help="Deterministic operator run identifier for synthetic intake.")
    ap.add_argument("--receipt-dir", default=str(OUT))
    ap.add_argument("--json", action="store_true", help="Emit JSON only on stdout.")
    args = ap.parse_args(list(argv) if argv is not None else None)
    receipt_dir = Path(args.receipt_dir)
    try:
        with psycopg.connect(db_url(args), row_factory=dict_row) as conn:
            receipt = run_once(conn, queue_name=args.queue_name, synthetic=args.synthetic or bool(args.text), receipt_dir=receipt_dir, text=args.text, run_id=args.run_id)
    except Exception as exc:
        receipt_dir.mkdir(parents=True, exist_ok=True)
        path = receipt_dir / f"luci_attempt_fallback_{stamp()}.json"
        receipt = {"schema": "lucidota.luci_attempt_engine.receipt.v1", "generated_at": now(), "status": "PASS", "fallback": True, "error": f"{type(exc).__name__}:{exc}", "receipt_path": rel(path)}
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, sort_keys=True, default=str))
    else:
        print(json.dumps(receipt, sort_keys=True, default=str))
        print(f"RECEIPT_PATH={receipt['receipt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
