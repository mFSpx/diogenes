#!/usr/bin/env python3
"""Enqueue bounded BGE embed-fill jobs for remaining NULL corpus chunks.

Jobs intentionally use selection=next_null instead of OFFSET. OFFSET over a
shrinking WHERE embedding IS NULL set skips work as chunks are filled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from math import ceil
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lucidota_ingestion_quality_audit import embedding_quality_sql_where

STATE_DSN = os.getenv("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")
STORAGE_DSN = os.getenv("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
DEFAULT_BATCH_SIZE = int(os.getenv("LUCIDOTA_EMBED_FILL_BATCH_SIZE", "500"))
WORKER_KEY = "embed_fill_batch"
QUEUE_NAME = "korpus"
OUT = ROOT / "05_OUTPUTS" / "embedding_enqueue"
SCHEMA = "lucidota.embed_fill_enqueuer.v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def build_job_rows(total_null: int, batch_size: int, max_jobs: int = 0) -> list[tuple[str, str]]:
    if total_null <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_jobs < 0:
        raise ValueError("max_jobs must be non-negative")
    rows: list[tuple[str, str]] = []
    planned = ceil(total_null / batch_size)
    if max_jobs:
        planned = min(planned, max_jobs)
    for batch_index in range(planned):
        idempotency_key = f"embed_fill_batch:next_null:v3_quality:{batch_size}:{batch_index}"
        payload = {
            "job_kind": WORKER_KEY,
            "selection": "next_null",
            "quality_gate": "readable_text_only",
            "limit": batch_size,
            "batch_index": batch_index,
        }
        rows.append((idempotency_key, json.dumps(payload, sort_keys=True, separators=(",", ":"))))
    return rows


def ensure_worker_contract(conn) -> None:
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lucidota_control.absurd_worker_contract
              (worker_key, queue_name, script_path, input_contract, output_contract,
               idempotency_rule, retry_policy, dead_letter_policy,
               canonical_graph_write_allowed, status, evidence_refs)
            VALUES ('embed_fill_batch','korpus','scripts/corpus_embed_fill_worker.py',
              '{"job_kind":"embed_fill_batch","required_fields":["limit"],"quality_gate":"readable_text_only"}',
              '{"receipt_glob":"05_OUTPUTS/runtime/embed_fill*.json"}',
              'next_null_quality_v3_batch_index','{"max_attempts":3,"backoff":"exponential"}',
              '{"policy":"quarantine"}', FALSE, 'implemented', '[]')
            ON CONFLICT (worker_key) DO UPDATE
            SET script_path=EXCLUDED.script_path,
                input_contract=EXCLUDED.input_contract,
                output_contract=EXCLUDED.output_contract,
                idempotency_rule=EXCLUDED.idempotency_rule,
                retry_policy=EXCLUDED.retry_policy,
                dead_letter_policy=EXCLUDED.dead_letter_policy,
                canonical_graph_write_allowed=EXCLUDED.canonical_graph_write_allowed,
                status=EXCLUDED.status
            """
        )


def retire_legacy_offset_jobs(conn) -> int:
    """Cancel old OFFSET/ungated jobs so bounded quality-gated jobs own the backlog."""
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE lucidota_control.absurd_queue_job
            SET status='cancelled',
                result=jsonb_build_object(
                    'reason', 'legacy_offset_embed_fill_retired',
                    'replaced_by', 'embed_fill_batch:next_null:v3_quality',
                    'retired_at', now()
                ),
                completed_at=now(),
                updated_at=now()
            WHERE queue_name=%s
              AND job_kind=%s
              AND status='queued'
              AND (
                payload ? 'offset'
                OR idempotency_key NOT LIKE 'embed_fill_batch:next_null:v3_quality:%%'
              )
            """,
            (QUEUE_NAME, WORKER_KEY),
        )
        return int(cur.rowcount)


def count_null_embeddings() -> int:
    conn = psycopg2.connect(STORAGE_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM lucidota_korpus.corpus_chunk WHERE {embedding_quality_sql_where()}")
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def write_receipt(report: dict, receipt_dir: Path | str = OUT) -> str:
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"embed_fill_enqueuer_{stamp()}.json"
    report["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report["receipt_path"]


def enqueue(batch_size: int, *, max_jobs: int = 0, dry_run: bool = False, receipt_dir: Path | str = OUT) -> dict:
    total_null = count_null_embeddings()
    rows = build_job_rows(total_null, batch_size, max_jobs=max_jobs)
    jobs_enqueued = 0
    legacy_jobs_retired = 0
    if not dry_run:
        conn = psycopg2.connect(STATE_DSN)
        try:
            ensure_worker_contract(conn)
            legacy_jobs_retired = retire_legacy_offset_jobs(conn)
            with conn, conn.cursor() as cur:
                for idempotency_key, payload in rows:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.absurd_queue_job
                          (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts)
                        VALUES ('korpus','embed-fill-pipeline','embed_fill_batch',%s,%s::jsonb,50,3)
                        ON CONFLICT (queue_name, idempotency_key) DO NOTHING
                        """,
                        (idempotency_key, payload),
                    )
                    if cur.rowcount > 0:
                        jobs_enqueued += 1
            conn.commit()
        finally:
            conn.close()
    report = {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "status": "PASS",
        "dry_run": dry_run,
        "total_null": total_null,
        "jobs_planned": len(rows),
        "jobs_enqueued": jobs_enqueued,
        "batch_size": batch_size,
        "max_jobs": max_jobs,
        "legacy_jobs_retired": legacy_jobs_retired,
        "queue_name": QUEUE_NAME,
        "worker_key": WORKER_KEY,
        "canonical_graph_writes_performed": False,
        "planned_jobs": [
            {"idempotency_key": idempotency_key, "payload": json.loads(payload)}
            for idempotency_key, payload in rows[:20]
        ],
    }
    write_receipt(report, receipt_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Enqueue bounded next-null BGE embed-fill jobs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-jobs", type=int, default=0, help="Cap jobs planned/enqueued; 0 means all current backlog batches")
    parser.add_argument("--dry-run", action="store_true", help="Plan and receipt jobs without writing queue rows")
    parser.add_argument("--receipt-dir", default=str(OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = enqueue(args.batch_size, max_jobs=args.max_jobs, dry_run=args.dry_run, receipt_dir=args.receipt_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(
            f"total_null={report['total_null']} jobs_planned={report['jobs_planned']} "
            f"jobs_enqueued={report['jobs_enqueued']} batch_size={report['batch_size']} "
            f"legacy_jobs_retired={report['legacy_jobs_retired']} receipt_path={report['receipt_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
