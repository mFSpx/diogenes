#!/usr/bin/env python3
"""corpus_embed_fill_worker.py — ABSURD consumer for embed_fill_batch jobs.

Mutation class: candidate_writer
Dequeues embed_fill_batch jobs from lucidota_control.absurd_queue_job (queue='korpus'),
embeds corpus_chunk rows via BGE fleet, writes embeddings back, emits receipt.

Worker contract: worker_key='embed_fill_batch', queue_name='korpus'
Enqueuer:        scripts/absurd_embed_fill_enqueuer.py

Usage:
    source scripts/lucidota_safe_ops_env.sh
    python3 scripts/corpus_embed_fill_worker.py [--concurrency 14] [--http-batch 32] [--dry-run]
    python3 scripts/corpus_embed_fill_worker.py --loop          # run until queue empty
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

STATE_DSN   = os.environ.get("LUCIDOTA_GO_STATE_DSN",   "postgresql:///lucidota_state")
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
WORKER_KEY  = "embed_fill_batch"
QUEUE_NAME  = "korpus"
OUT_DIR     = ROOT / "05_OUTPUTS" / "runtime"

BGE_ENDPOINTS = [
    e.strip()
    for e in os.environ.get("LUCIDOTA_BGE_FLEET", "").split(",")
    if e.strip()
] or ["http://127.0.0.1:8101"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── ABSURD dequeue ─────────────────────────────────────────────────────────────

def dequeue_one(conn: psycopg2.extensions.connection) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            UPDATE lucidota_control.absurd_queue_job
            SET    status         = 'running',
                   attempt_count  = attempt_count + 1,
                   leased_by      = %s,
                   lease_expires_at = now() + interval '30 minutes',
                   updated_at     = now()
            WHERE  job_uuid = (
                SELECT job_uuid
                FROM   lucidota_control.absurd_queue_job
                WHERE  queue_name = %s
                  AND  job_kind   = %s
                  AND  status     = 'queued'
                  AND  run_after <= now()
                ORDER  BY priority ASC, created_at ASC
                FOR    UPDATE SKIP LOCKED
                LIMIT  1
            )
            RETURNING *
        """, (WORKER_KEY, QUEUE_NAME, WORKER_KEY))
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None


def mark_done(conn, job_uuid: str, result: dict) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE lucidota_control.absurd_queue_job
            SET status='succeeded', result=%s::jsonb, completed_at=now(), updated_at=now()
            WHERE job_uuid=%s
        """, (json.dumps(result), job_uuid))
        conn.commit()


def mark_failed(conn, job_uuid: str, error: str, attempt: int, max_attempts: int) -> None:
    next_status = "queued" if attempt < max_attempts else "dead_lettered"
    backoff = min(60 * (2 ** attempt), 3600)
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE lucidota_control.absurd_queue_job
            SET status=%s, last_error=%s,
                run_after = now() + %s * interval '1 second',
                updated_at = now()
            WHERE job_uuid=%s
        """, (next_status, error[:2000], backoff, job_uuid))
        conn.commit()


# ── BGE async embed ─────────────────────────────────────────────────────────────

async def embed_http_batch(
    session: aiohttp.ClientSession,
    texts: list[str],
    timeout: int = 60,
) -> list[list[float]] | None:
    url = random.choice(BGE_ENDPOINTS).rstrip("/") + "/v1/embeddings"
    try:
        async with session.post(
            url,
            json={"model": "bge-m3", "input": texts},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"[embed_worker] HTTP {resp.status}: {body[:200]}", file=sys.stderr)
                return None
            data = await resp.json()
            items = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [d["embedding"] for d in items]
    except Exception as e:
        print(f"[embed_worker] request error {type(e).__name__}: {e!r}", file=sys.stderr)
        return None


async def embed_all_async(
    rows: list[tuple],
    http_batch: int,
    concurrency: int,
) -> tuple[list[tuple], int]:
    """Embed all rows concurrently. Returns ([(uuid, vec), ...], error_count)."""
    sem = asyncio.Semaphore(concurrency)
    results: list[tuple | None] = [None] * len(rows)
    errors = 0

    async def process_slice(start: int, end: int) -> None:
        nonlocal errors
        slice_rows = rows[start:end]
        texts = [r[1] for r in slice_rows]
        uuids = [r[0] for r in slice_rows]
        async with sem:
            vecs = await embed_http_batch(session, texts)
        if vecs is None or len(vecs) != len(texts):
            errors += len(slice_rows)
            return
        for i, (uid, vec) in enumerate(zip(uuids, vecs)):
            results[start + i] = (uid, vec)

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(process_slice(i, min(i + http_batch, len(rows))))
            for i in range(0, len(rows), http_batch)
        ]
        await asyncio.gather(*tasks)

    good = [(uid, vec) for r in results if r is not None for uid, vec in [r]]
    return good, errors


# ── DB write ────────────────────────────────────────────────────────────────────

def write_embeddings(pairs: list[tuple]) -> int:
    if not pairs:
        return 0
    conn = psycopg2.connect(STORAGE_DSN)
    try:
        update_data = [
            ("[" + ",".join(repr(float(x)) for x in vec) + "]", "bge-m3", uid)
            for uid, vec in pairs
        ]
        with conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """UPDATE lucidota_korpus.corpus_chunk
                   SET embedding = %s::vector, embedding_model = %s,
                       extractor = 'corpus_embed_fill_worker'
                   WHERE chunk_uuid = %s AND embedding IS NULL""",
                update_data,
                page_size=min(len(update_data), 500),
            )
        return len(pairs)
    finally:
        conn.close()


# ── Job handler ─────────────────────────────────────────────────────────────────

def run_job(
    job: dict,
    concurrency: int,
    http_batch: int,
    dry_run: bool,
    max_chunks: int = 0,
) -> dict:
    payload = job["payload"]
    offset  = int(payload["offset"])
    limit   = int(payload["limit"])
    if max_chunks > 0:
        limit = min(limit, max_chunks)

    # Fetch rows
    conn = psycopg2.connect(STORAGE_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_uuid::text, content FROM lucidota_korpus.corpus_chunk"
                " WHERE embedding IS NULL ORDER BY created_at OFFSET %s LIMIT %s",
                (offset, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"filled": 0, "errors": 0, "skipped": limit, "note": "no null rows at offset"}

    t0 = time.time()
    if dry_run:
        elapsed = time.time() - t0
        return {"filled": 0, "errors": 0, "dry_run": True, "rows_found": len(rows)}

    pairs, errors = asyncio.run(embed_all_async(rows, http_batch, concurrency))

    filled = write_embeddings(pairs)
    elapsed = time.time() - t0
    rate = filled / elapsed if elapsed > 0 else 0

    return {
        "filled":    filled,
        "errors":    errors,
        "elapsed_s": round(elapsed, 2),
        "rate_per_s": round(rate, 2),
        "offset":    offset,
        "limit":     limit,
    }


# ── Receipt ─────────────────────────────────────────────────────────────────────

def write_receipt(job_uuid: str, result: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"embed_fill_{_stamp()}_{job_uuid[:8]}.json"
    receipt = {
        "receipt_type":  "ABSURD_POSTGRES_RUNTIME",
        "worker_key":    WORKER_KEY,
        "job_uuid":      job_uuid,
        "completed_at":  _now_iso(),
        "result":        result,
    }
    path.write_text(json.dumps(receipt, indent=2))
    return path


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="ABSURD embed_fill_batch consumer")
    ap.add_argument("--concurrency", type=int, default=14,
                    help="Parallel HTTP requests to BGE (default 14; leave 2 of 16 slots free)")
    ap.add_argument("--http-batch", type=int, default=32,
                    help="Texts per HTTP POST to /v1/embeddings (default 32)")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="Cap rows per job for testing (0=full job)")
    ap.add_argument("--loop", action="store_true",
                    help="Keep running until queue is empty")
    ap.add_argument("--dry-run", action="store_true",
                    help="Claim job, fetch rows, skip HTTP + DB writes")
    args = ap.parse_args()

    conn = psycopg2.connect(STATE_DSN)
    jobs_done = 0

    try:
        while True:
            job = dequeue_one(conn)
            if job is None:
                if jobs_done == 0:
                    print("[embed_worker] no queued embed_fill_batch jobs found")
                else:
                    print(f"[embed_worker] queue empty after {jobs_done} jobs")
                break

            job_uuid = str(job["job_uuid"])
            attempt  = job["attempt_count"]
            max_att  = job["max_attempts"]
            print(f"[embed_worker] claimed job {job_uuid[:8]} attempt={attempt} "
                  f"offset={job['payload'].get('offset')} limit={job['payload'].get('limit')}")

            try:
                result = run_job(job, args.concurrency, args.http_batch, args.dry_run,
                                 max_chunks=args.max_chunks)
                mark_done(conn, job_uuid, result)
                receipt_path = write_receipt(job_uuid, result)
                jobs_done += 1
                print(f"[embed_worker] done job={job_uuid[:8]} filled={result.get('filled')} "
                      f"errors={result.get('errors')} rate={result.get('rate_per_s')}/s "
                      f"receipt={receipt_path.name}")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[embed_worker] ERROR job={job_uuid[:8]}: {err}", file=sys.stderr)
                conn.rollback()
                mark_failed(conn, job_uuid, err, attempt, max_att)

            if not args.loop:
                break

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
