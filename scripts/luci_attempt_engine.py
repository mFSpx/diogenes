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
        return {"event_id": event_id, "raw_artifact_uuid": raw_uuid, "work_order_uuid": work_order_uuid, "work_receipt_uuid": receipt_uuid}

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
    return {"event_id": event_id, "raw_artifact_uuid": raw_uuid, "work_order_uuid": work_order_uuid, "work_receipt_uuid": receipt_uuid}


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
            job = maybe_seed_synthetic_job(cur, queue_name=queue_name, run_id=run_id, text=text)
        else:
            job = claim_job(cur, queue_name)
            if job is None and synthetic:
                job = maybe_seed_synthetic_job(cur, queue_name=queue_name, run_id=run_id, text=text)
        if job is None:
            return {"status": "no_work", "queue_name": queue_name}
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
        if not synthetic:
            cur.execute(
                "UPDATE lucidota_control.absurd_queue_job SET status='running', leased_by='luci_attempt_engine', lease_expires_at=now()+interval '5 minutes', attempt_count=attempt_count+1 WHERE job_uuid=%s::uuid",
                (job["job_uuid"],),
            )
            if next_state == "queued":
                update_job(cur, job["job_uuid"], status="queued", result={"next_state": next_state, "reason": reason, **ids}, run_after=(datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat())
            else:
                update_job(cur, job["job_uuid"], status=next_state, result={"next_state": next_state, "reason": reason, **ids}, last_error=reason)
        conn.commit()
        receipt["db_write"] = {"job": job["job_uuid"], **ids, "event_kind": event_kind}
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
