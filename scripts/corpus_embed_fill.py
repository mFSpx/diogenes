#!/usr/bin/env python3
"""corpus_embed_fill.py — batch-fill NULL embeddings in corpus_chunk.

Runs after a LUCIDOTA_SKIP_EMBED=1 crush pass. Reads rows where
embedding IS NULL, batches them (default 32 per HTTP call), embeds via
BGE fleet, writes back. Much faster than single-chunk embedding:
  - Single mode: 248ms/chunk × 1 = 248ms
  - Batch mode: ~280ms for 32 chunks = 8.75ms/chunk = ~28x speedup

Usage:
    source scripts/lucidota_safe_ops_env.sh
    python3 scripts/corpus_embed_fill.py [--batch 32] [--workers 4] [--limit N] [--dry-run]
"""
from __future__ import annotations
import argparse, os, sys, time, threading, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import requests
import psycopg2, psycopg2.extras

STORAGE_DSN = os.environ.get('LUCIDOTA_GO_STORAGE_DSN', 'postgresql:///lucidota_storage')
BGE_ENDPOINTS = [e.strip() for e in os.environ.get('LUCIDOTA_BGE_FLEET', '').split(',') if e.strip()] or [
    'http://127.0.0.1:8101', 'http://127.0.0.1:8102',
    'http://127.0.0.1:8103', 'http://127.0.0.1:8104',
]

_ctr_lock = threading.Lock()
_done = 0
_errors = 0


def embed_batch(chunks: list[str], timeout: int = 120) -> list[list[float]] | None:
    base = random.choice(BGE_ENDPOINTS).rstrip('/')
    url = f"{base}/v1/embeddings"
    try:
        r = requests.post(url, json={'model': 'bge-m3', 'input': chunks},
                          headers={'User-Agent': 'groq-python/0.28.0'}, timeout=timeout)
        if r.status_code == 200:
            data = r.json()['data']
            return [d['embedding'] for d in sorted(data, key=lambda x: x.get('index', 0))]
    except Exception as e:
        print(f'[embed_fill] batch embed error: {e}', file=sys.stderr)
    return None


def process_batch(rows: list[tuple], dry_run: bool) -> int:
    global _done, _errors
    if not rows:
        return 0
    chunks = [r[1] for r in rows]
    uuids = [r[0] for r in rows]

    embeddings = embed_batch(chunks)
    if embeddings is None or len(embeddings) != len(chunks):
        with _ctr_lock:
            _errors += len(rows)
        return 0

    if dry_run:
        with _ctr_lock:
            _done += len(rows)
        return len(rows)

    conn = psycopg2.connect(STORAGE_DSN)
    try:
        update_data = [
            ('[' + ','.join(repr(float(x)) for x in emb) + ']', 'bge-m3', uid)
            for uid, emb in zip(uuids, embeddings)
        ]
        with conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                UPDATE lucidota_korpus.corpus_chunk
                SET embedding = %s::vector, embedding_model = %s, extractor = 'corpus_embed_fill'
                WHERE chunk_uuid = %s AND embedding IS NULL
            """, update_data, page_size=len(update_data))
        with _ctr_lock:
            _done += len(rows)
        return len(rows)
    except Exception as e:
        print(f'[embed_fill] DB write error: {e}', file=sys.stderr)
        with _ctr_lock:
            _errors += len(rows)
        return 0
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description='Batch-fill NULL embeddings in corpus_chunk')
    parser.add_argument('--batch', type=int, default=32, help='Chunks per BGE call (default 32)')
    parser.add_argument('--workers', type=int, default=4, help='Parallel fill threads (default 4)')
    parser.add_argument('--limit', type=int, default=0, help='Max chunks to fill (0=all)')
    parser.add_argument('--dry-run', action='store_true', help='Count only, no writes')
    args = parser.parse_args()

    conn = psycopg2.connect(STORAGE_DSN)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM lucidota_korpus.corpus_chunk WHERE embedding IS NULL")
        total_null = cur.fetchone()[0]
    conn.close()
    print(f'[embed_fill] NULL embeddings to fill: {total_null}')
    if total_null == 0:
        print('[embed_fill] Nothing to do.')
        return 0

    limit = args.limit if args.limit > 0 else total_null
    t0 = time.time()

    conn = psycopg2.connect(STORAGE_DSN)
    with conn.cursor('fill_cursor') as cur:
        cur.execute(
            "SELECT chunk_uuid, content FROM lucidota_korpus.corpus_chunk "
            "WHERE embedding IS NULL ORDER BY created_at LIMIT %s",
            (limit,)
        )

        from concurrent.futures import ThreadPoolExecutor
        batch = []
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                row = cur.fetchone()
                if row is None:
                    if batch:
                        futures.append(pool.submit(process_batch, list(batch), args.dry_run))
                    break
                batch.append(row)
                if len(batch) >= args.batch:
                    futures.append(pool.submit(process_batch, list(batch), args.dry_run))
                    batch = []
            for f in futures:
                f.result()
    conn.close()

    elapsed = time.time() - t0
    rate = _done / elapsed if elapsed > 0 else 0
    print(f'[embed_fill] done={_done} errors={_errors} elapsed={elapsed:.1f}s rate={rate:.1f}/s')
    print(f'[receipt] corpus_embed_fill filled={_done} dry_run={args.dry_run}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
