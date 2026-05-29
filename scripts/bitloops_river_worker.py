#!/usr/bin/env python3
"""ABSURD worker: consume bitloops_momentary queue, feed River HoeffdingTree.

Mutation class: candidate_writer (bitloops_loop.checkpoint_event only)
Harvest mode: labels are always True until real outcome signals arrive.
Never writes canonical graph tables.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from absurd_worker_contracts import validate_worker_contract

OUT_DIR = ROOT / "05_OUTPUTS" / "receipts"
QUEUE = "bitloops_momentary"
JOB_KIND = "bitloops_context_ingest"
WORKER_KEY = "bitloops_river_worker"
DSN = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_receipt(name: str, data: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}_{_stamp()}.json"
    p.write_text(json.dumps(data, indent=2) + "\n")
    return p


def _make_model():
    try:
        from river.tree import HoeffdingTreeClassifier
        return HoeffdingTreeClassifier()
    except ImportError:
        return None


def _extract_features(envelope: dict) -> dict:
    query = envelope.get("devql_query") or ""
    route = envelope.get("ontology_route") or ""
    return {
        "devql_query_len": len(query),
        "ontology_route_hash": hash(route) % 1000,
        "hour_of_day": datetime.now(timezone.utc).hour,
    }


def process_one(conn: psycopg.Connection, model) -> bool:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT j.job_uuid, j.queue_name, j.job_kind, j.payload, j.workflow_name, j.idempotency_key
            FROM lucidota_control.absurd_queue_job j
            WHERE j.queue_name = %s AND j.status = 'queued'
            ORDER BY j.created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """,
            (QUEUE,),
        )
        job = cur.fetchone()
        if job is None:
            return False

        decision = validate_worker_contract(cur, queue_name=QUEUE, job_kind=job["job_kind"], worker_key=WORKER_KEY)
        if not decision.ok:
            cur.execute(
                "UPDATE lucidota_control.absurd_queue_job SET status='dead_lettered', updated_at=now() WHERE job_uuid=%s",
                (job["job_uuid"],),
            )
            conn.commit()
            return False

        payload = job["payload"] if isinstance(job["payload"], dict) else json.loads(job["payload"] or "{}")
        envelope = payload if "checkpoint_id" in payload else payload.get("envelope", payload)

        features = _extract_features(envelope)
        if model is not None:
            model.learn_one(features, True)  # harvest mode: label=True until real outcomes

        # Delta seconds between consecutive examples for this session
        delta_seconds = None
        session_id = envelope.get("session_id", "")
        cur.execute(
            """
            SELECT processed_at FROM bitloops_loop.checkpoint_event
            WHERE session_id = %s AND processed_at IS NOT NULL
            ORDER BY processed_at DESC LIMIT 1
            """,
            (session_id,),
        )
        prev = cur.fetchone()
        if prev and prev["processed_at"]:
            delta_seconds = (datetime.now(timezone.utc) - prev["processed_at"].replace(tzinfo=timezone.utc)).total_seconds()

        now = datetime.now(timezone.utc)
        cur.execute(
            """
            INSERT INTO bitloops_loop.checkpoint_event
              (checkpoint_id, session_id, context_blob_hash, devql_query, ontology_route,
               timestamp_custody, workflow_id, idempotency_key, processed_at, river_correct, river_delta_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO UPDATE
              SET processed_at = EXCLUDED.processed_at,
                  river_delta_seconds = EXCLUDED.river_delta_seconds
            """,
            (
                envelope.get("checkpoint_id", ""),
                session_id,
                envelope.get("context_blob_hash"),
                envelope.get("devql_query"),
                envelope.get("ontology_route"),
                envelope.get("timestamp_custody", now.isoformat()),
                envelope.get("workflow_id", "bitloops_automation_loop"),
                envelope.get("idempotency_key", envelope.get("checkpoint_id", "")),
                now,
                True,  # harvest
                delta_seconds,
            ),
        )

        cur.execute(
            "UPDATE lucidota_control.absurd_queue_job SET status='succeeded', updated_at=now() WHERE job_uuid=%s",
            (job["job_uuid"],),
        )
        conn.commit()

        p = _write_receipt("bitloops_river", {
            "schema": "lucidota.bitloops_loop.river.v1",
            "job_uuid": str(job["job_uuid"]),
            "checkpoint_id": envelope.get("checkpoint_id"),
            "session_id": session_id,
            "features": features,
            "river_correct": True,
            "river_delta_seconds": delta_seconds,
            "generated_at": now.isoformat(),
        })
        print(f"RECEIPT_PATH={p}")
        print("BITLOOPS_RIVER=PASS")
        return True


def main() -> int:
    model = _make_model()
    with psycopg.connect(DSN) as conn:
        found = process_one(conn, model)
    if not found:
        print("BITLOOPS_RIVER=IDLE no queued jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
