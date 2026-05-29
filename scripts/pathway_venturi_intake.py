#!/usr/bin/env python3
"""
Pathway Venturi Intake — KRAMPUSCHEWING corpus pipe.

NEEDLE THROUGHPUT: 2000 tokens/second per endpoint. Six endpoints = 12,000 tok/s.
DO NOT rate-limit, throttle, or add sleeps to Needle calls. They are the fastest
thing in the stack. Hammer them.

Architecture:
  KRAMPUSCHEWING (files) → [Pathway Rust engine]
    → magic bytes routing → text extraction
    → chunk (400-word windows, Needle sweet spot)
    → async fan-out to :8090-:8095 (6x Needle @ 2000 tok/s each)
    → GO-25 candidate wrapping
    → raw_artifact dedup record (lucidota_state)
    → absurd_queue_job enqueue (queue='venturi_intake', job_kind='go25_stage')

mutation_class: candidate_writer (writes staging packets + ABSURD jobs only)
NO CANONICAL GRAPH WRITES. Promotion gate handles truth.

Usage:
  python3 scripts/pathway_venturi_intake.py
  # or as daemon:
  python3 scripts/pathway_venturi_intake.py --corpus-dir KRAMPUSCHEWING

Dependencies:
  pip install pathway python-magic pdfminer.six psycopg httpx

Pathway docs: https://pathway.com/developers
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ── Configuration ─────────────────────────────────────────────────────────────
CORPUS_DIR = os.environ.get("VENTURI_SOURCE_DIR", str(ROOT / "KRAMPUSCHEWING"))
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
QUEUE_NAME = "venturi_intake"
WORKFLOW_NAME = "pathway-venturi-intake"

# NEEDLES: 2000 tokens/second each. Six endpoints. RUN THEM HARD.
# These 26M models are the fastest text-to-GO25 path in the stack.
# Do NOT add rate limiting here. Do NOT add sleeps. Full throttle.
NEEDLE_ENDPOINTS = [
    f"http://127.0.0.1:{p}/v1"
    for p in range(8090, 8096)
]
NEEDLE_MODEL = "needle"
NEEDLE_MAX_TOKENS = 512
NEEDLE_TIMEOUT_SEC = 8.0  # 8s = ~16,000 tok budget per call at 2k tok/s

# Extraction config
CHUNK_WORDS = 400           # Needle sweet spot
MAX_CHUNKS_PER_FILE = 6     # one per Needle endpoint per pass
MAX_FILE_BYTES = 50_000_000 # 50MB hard cap per file pass

ACTOR = f"venturi@{socket.gethostname()}"


# ── Utility ────────────────────────────────────────────────────────────────────

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def job_idem_key(path: str, content_hash: str) -> str:
    return sha256_hex(f"{path}|{content_hash}".encode())[:48]


# ── Text Extraction ────────────────────────────────────────────────────────────

def detect_mime(raw: bytes) -> str:
    try:
        import magic
        return magic.from_buffer(raw[:2048], mime=True)
    except ImportError:
        # fallback: header sniff
        if raw[:4] == b"%PDF":
            return "application/pdf"
        if raw[:2] in (b"PK",):
            return "application/zip"
        return "text/plain"


def extract_text(file_path: str, raw: bytes) -> str:
    mime = detect_mime(raw)

    if mime == "application/pdf":
        try:
            from pdfminer.high_level import extract_text as pdfmine
            from io import BytesIO
            return pdfmine(BytesIO(raw)) or ""
        except Exception as e:
            return f"[PDF_EXTRACT_ERROR: {e}]"

    if mime in ("application/zip", "application/x-zip-compressed"):
        # Top-level zip: index the filenames for now; Rust exarch handles full unpack
        try:
            import zipfile
            from io import BytesIO
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                names = zf.namelist()
                return f"[ZIP_MANIFEST: {len(names)} files]\n" + "\n".join(names[:200])
        except Exception as e:
            return f"[ZIP_ERROR: {e}]"

    if mime.startswith("image/"):
        # SmolDocling OCR endpoint — wire when :8086 is live
        return f"[IMAGE_OCR_PENDING: {Path(file_path).name} | {len(raw)} bytes]"

    if mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        try:
            from docx import Document
            from io import BytesIO
            doc = Document(BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return raw.decode("utf-8", errors="replace")[:8192]

    # Default: UTF-8 best-effort
    return raw.decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i : i + chunk_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks[:MAX_CHUNKS_PER_FILE]


# ── Needle Fan-Out ─────────────────────────────────────────────────────────────
# NEEDLE THROUGHPUT IS 2000 tok/s PER ENDPOINT.
# Six endpoints = 12,000 tok/s combined.
# These 26M models are not bottlenecks. The disk and network are.
# Do not add rate limiting. Do not add sleeps. Run them simultaneously.

async def call_one_needle(endpoint: str, chunk: str, client: Any) -> dict:
    """Call a single Needle endpoint. No retry — fast fail, next file wins."""
    try:
        resp = await client.post(
            f"{endpoint}/chat/completions",
            json={
                "model": NEEDLE_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a GO-25 extraction engine. "
                            "Return ONLY a JSON object with keys: "
                            '"objects" (list), "events" (list), "edges" (list). '
                            "Each object has: type, label, properties. "
                            "Each event has: type, label, timestamp_hint. "
                            "Each edge has: from_label, to_label, relation. "
                            "No markdown. No explanation. JSON only."
                        ),
                    },
                    {"role": "user", "content": chunk},
                ],
                "max_tokens": NEEDLE_MAX_TOKENS,
                "temperature": 0.0,
            },
            timeout=NEEDLE_TIMEOUT_SEC,
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present (Forge proxy handles this upstream,
        # but guard here too for direct calls)
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                l for l in lines if not l.startswith("```")
            )
        return json.loads(content)
    except Exception:
        return {}


async def needle_swarm_fanout(chunks: list[str]) -> list[dict]:
    """
    Fan all chunks out to Needle endpoints simultaneously.
    2000 tok/s per endpoint — no throttle, no sleep, full blast.
    """
    import asyncio
    import httpx

    async with httpx.AsyncClient() as client:
        tasks = [
            call_one_needle(
                NEEDLE_ENDPOINTS[i % len(NEEDLE_ENDPOINTS)],
                chunk,
                client,
            )
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r]


# ── GO-25 Staging Packet ───────────────────────────────────────────────────────

def build_go25_packet(
    file_path: str,
    content_hash: str,
    needle_results: list[dict],
    mime_type: str,
) -> dict:
    return {
        "schema": "go25_staging_packet_v1",
        "mutation_class": "candidate_writer",
        "source_path": file_path,
        "content_hash": content_hash,
        "mime_type": mime_type,
        "objects": [o for r in needle_results for o in r.get("objects", [])],
        "events": [e for r in needle_results for e in r.get("events", [])],
        "edges": [ed for r in needle_results for ed in r.get("edges", [])],
        "needle_endpoint_count": len(NEEDLE_ENDPOINTS),
        "chunks_processed": len(needle_results),
        "staged_at": now_z(),
        "promoted": False,
        "actor": ACTOR,
    }


# ── Postgres Writers ────────────────────────────────────────────────────────────

def record_raw_artifact(cur: Any, path: str, content_hash: str, mime: str, byte_count: int) -> str:
    """Write dedup record to raw_artifact. Returns 'inserted' or 'exists'."""
    cur.execute(
        """
        INSERT INTO lucidota_control.raw_artifact
            (raw_ref, raw_sha256, hash_algo, source, actor, byte_count,
             char_count, mime_type, storage_hint, detail)
        VALUES (%s, %s, 'sha256', 'venturi_intake', %s, %s, 0, %s,
                'cas_path', %s::jsonb)
        ON CONFLICT DO NOTHING
        """,
        (
            path,
            content_hash,
            ACTOR,
            byte_count,
            mime,
            json.dumps({"source_dir": CORPUS_DIR}),
        ),
    )
    return "inserted" if cur.rowcount else "exists"


def enqueue_go25_stage(cur: Any, idem_key: str, packet: dict) -> str:
    """Write GO-25 staging packet to ABSURD queue. Returns job_uuid."""
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_job
            (queue_name, workflow_name, job_kind, idempotency_key, payload)
        VALUES (%s, %s, 'go25_stage', %s, %s::jsonb)
        ON CONFLICT (queue_name, idempotency_key) DO NOTHING
        RETURNING job_uuid::text
        """,
        (
            QUEUE_NAME,
            WORKFLOW_NAME,
            idem_key,
            json.dumps(packet),
        ),
    )
    row = cur.fetchone()
    return row[0] if row else "duplicate"


# ── Per-File Processing ─────────────────────────────────────────────────────────

def process_file_sync(file_path: str, raw_bytes: bytes) -> dict:
    """
    Synchronous wrapper called by Pathway's pw.apply().
    Returns a status dict for the sink.
    """
    import asyncio

    path = str(file_path)
    if len(raw_bytes) > MAX_FILE_BYTES:
        raw_bytes = raw_bytes[:MAX_FILE_BYTES]

    content_hash = sha256_hex(raw_bytes)
    mime = detect_mime(raw_bytes)
    text = extract_text(path, raw_bytes)
    chunks = chunk_text(text)

    needle_results: list[dict] = []
    if chunks:
        try:
            needle_results = asyncio.run(needle_swarm_fanout(chunks))
        except Exception:
            pass  # Needles may be offline — still record the artifact

    packet = build_go25_packet(path, content_hash, needle_results, mime)
    idem_key = job_idem_key(path, content_hash)

    try:
        import psycopg
        with psycopg.connect(STATE_DSN) as conn:
            with conn.cursor() as cur:
                artifact_status = record_raw_artifact(
                    cur, path, content_hash, mime, len(raw_bytes)
                )
                if artifact_status == "exists":
                    return {
                        "file_path": path,
                        "status": "dedup_skip",
                        "content_hash": content_hash,
                    }
                job_uuid = enqueue_go25_stage(cur, idem_key, packet)
            conn.commit()
    except Exception as e:
        return {"file_path": path, "status": f"db_error:{e}", "content_hash": content_hash}

    return {
        "file_path": path,
        "status": "staged",
        "content_hash": content_hash,
        "job_uuid": job_uuid,
        "objects": len(packet["objects"]),
        "events": len(packet["events"]),
        "edges": len(packet["edges"]),
    }


# ── Pathway Pipeline ────────────────────────────────────────────────────────────

def build_pipeline(corpus_dir: str) -> None:
    """
    Construct and run the Pathway streaming pipeline.
    Never exits — runs as a daemon watching the corpus directory.
    """
    import pathway as pw

    print(f"[venturi] Starting Pathway pipeline")
    print(f"[venturi] Source: {corpus_dir}")
    print(f"[venturi] Needles: {NEEDLE_ENDPOINTS} @ 2000 tok/s each (DO NOT THROTTLE)")
    print(f"[venturi] State DB: {STATE_DSN}")
    print(f"[venturi] Queue: {QUEUE_NAME}")

    # Source: Pathway's Rust engine watches the directory for new/modified files.
    # It treats the filesystem as a live streaming Kafka topic.
    # Only the diff is reprocessed on modification — not the whole file.
    raw_files = pw.io.fs.read(
        corpus_dir,
        format="binary",
        mode="streaming",
        with_metadata=True,
    )

    # Process each file: extract → chunk → Needle fan-out → stage
    # pw.apply runs process_file_sync in Pathway's multithreaded Rust executor.
    # The GIL does not apply here — full CPU parallelism across all cores.
    results = raw_files.select(
        result=pw.apply(
            process_file_sync,
            pw.this._metadata["path"],
            pw.this.data,
        )
    )

    # Sink: log results to stdout (structured JSON per file)
    pw.io.jsonlines.write(results, "-")

    # Start the engine. This call never returns.
    # Pathway compiles the above Python into a Rust computation graph and runs it.
    pw.run()


# ── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pathway Venturi Intake — KRAMPUSCHEWING corpus pipe"
    )
    parser.add_argument(
        "--corpus-dir",
        default=CORPUS_DIR,
        help=f"Source directory to watch (default: {CORPUS_DIR})",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )
    args = parser.parse_args()

    if args.check_deps:
        missing = []
        for pkg in ["pathway", "magic", "pdfminer", "psycopg", "httpx"]:
            try:
                __import__(pkg.replace(".", "_").replace("-", "_"))
            except ImportError:
                missing.append(pkg)
        if missing:
            print(f"MISSING: {', '.join(missing)}")
            print("Install: pip install pathway python-magic pdfminer.six psycopg httpx")
            raise SystemExit(1)
        print("All deps present.")
        raise SystemExit(0)

    build_pipeline(args.corpus_dir)
