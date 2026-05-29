#!/usr/bin/env python3
"""Bounded read-only Chroma -> GLiNER claim-packet dry-run probe.

Reads Chroma SQLite with keyset pagination, never loads the full DB/table, and
runs the existing GLiNER claim-packet dry-run in bounded mini-batches.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "KRAMPUSCHEWING/Lucidota/Lucidota/CHROMADB/chroma.sqlite3"
OUT_ROOT = ROOT / "05_OUTPUTS/chroma_gliner_probe"


def now_z() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def iter_chunks(conn: sqlite3.Connection, *, limit: int, page_size: int, max_chars_per_chunk: int, start_after_id: int = 0) -> Iterator[dict[str, Any]]:
    emitted = 0
    last_id = start_after_id
    while emitted < limit:
        take = min(page_size, limit - emitted)
        rows = conn.execute(
            """
            SELECT e.id AS row_id,
                   e.embedding_id AS embedding_id,
                   doc.string_value AS text,
                   src.string_value AS source_file,
                   ci.int_value AS chunk_index
            FROM embeddings e
            JOIN embedding_metadata doc ON doc.id=e.id AND doc.key='chroma:document'
            LEFT JOIN embedding_metadata src ON src.id=e.id AND src.key='source_file'
            LEFT JOIN embedding_metadata ci ON ci.id=e.id AND ci.key='chunk_index'
            WHERE e.id > ?
            ORDER BY e.id ASC
            LIMIT ?
            """,
            (last_id, take),
        ).fetchall()
        if not rows:
            return
        for row in rows:
            last_id = int(row["row_id"])
            text = str(row["text"] or "")
            yield {
                "row_id": last_id,
                "embedding_id": row["embedding_id"],
                "source_file": row["source_file"],
                "chunk_index": row["chunk_index"],
                "text": text[:max_chars_per_chunk],
                "original_chars": len(text),
                "truncated": len(text) > max_chars_per_chunk,
            }
            emitted += 1
            if emitted >= limit:
                return


def run_reaped(cmd: list[str], *, timeout: int) -> dict[str, Any]:
    started = now_z()
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            stdout, stderr = proc.communicate()
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "started_at": started,
        "finished_at": now_z(),
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
        "reaped": True,
    }


def flush_batch(batch: list[dict[str, Any]], *, batch_no: int, out_dir: Path, dry_run_script: str, timeout: int) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")
    text_parts = []
    for item in batch:
        clean_text = str(item["text"]).replace("\x00", "\uFFFD")
        text_parts.append(f"\n\n[CHROMA row_id={item['row_id']} embedding_id={item['embedding_id']} source={item.get('source_file')} chunk={item.get('chunk_index')}]\n{clean_text}")
    text = "".join(text_parts)
    batch_dir = out_dir / f"batch_{batch_no:04d}"
    cmd = [
        sys.executable,
        dry_run_script,
        "--text",
        text,
        "--artifact-uuid",
        f"chroma-sqlite-probe-{batch_no:04d}",
        "--component-uuid",
        f"chroma-bounded-batch-{batch_no:04d}",
        "--out-dir",
        str(batch_dir),
    ]
    result = run_reaped(cmd, timeout=timeout)
    report_path = None
    for line in result["stdout_tail"].splitlines():
        if line.startswith("REPORT_PATH="):
            report_path = line.split("=", 1)[1]
    claim_count = None
    blockers: list[str] = []
    if report_path:
        try:
            d = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
            claim_count = d.get("claim_packet_count")
            blockers = d.get("blockers") or []
            result["extractor_backend"] = d.get("extractor_backend")
        except Exception as exc:
            blockers = [f"report_read_failed:{type(exc).__name__}"]
    return {
        "batch_no": batch_no,
        "chunk_count": len(batch),
        "text_chars": len(text),
        "first_row_id": batch[0]["row_id"],
        "last_row_id": batch[-1]["row_id"],
        "report_path": report_path,
        "claim_packet_count": claim_count,
        "blockers": blockers,
        "command": result,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Bounded Chroma SQLite to GLiNER dry-run probe")
    ap.add_argument("--sqlite", default=str(DEFAULT_DB))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--start-after-id", type=int, default=0)
    ap.add_argument("--max-chars-per-chunk", type=int, default=1600)
    ap.add_argument("--max-chars-per-run", type=int, default=60000)
    ap.add_argument("--command-timeout", type=int, default=120)
    ap.add_argument("--dry-run-script", default="scripts/gliner_claim_packet_dry_run.py")
    args = ap.parse_args()
    if args.limit < 1 or args.page_size < 1:
        raise SystemExit("limit/page-size must be positive")
    if args.page_size > args.limit:
        args.page_size = args.limit
    db = Path(args.sqlite)
    run_dir = OUT_ROOT / stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    batch: list[dict[str, Any]] = []
    batch_chars = 0
    batch_no = 0
    chunks_seen = 0
    original_chars = 0
    truncated_chunks = 0
    commands: list[dict[str, Any]] = []
    blockers: list[str] = []
    with connect_ro(db) as conn:
        total_chunks = int(conn.execute("SELECT count(*) FROM embeddings").fetchone()[0])
        for chunk in iter_chunks(conn, limit=args.limit, page_size=args.page_size, max_chars_per_chunk=args.max_chars_per_chunk, start_after_id=args.start_after_id):
            chunk_chars = len(chunk["text"]) + 160
            if batch and batch_chars + chunk_chars > args.max_chars_per_run:
                batch_no += 1
                res = flush_batch(batch, batch_no=batch_no, out_dir=run_dir, dry_run_script=args.dry_run_script, timeout=args.command_timeout)
                commands.append(res)
                if res["command"]["returncode"] != 0 or res["command"].get("timed_out"):
                    blockers.append(f"gliner_batch_failed:{batch_no}")
                batch = []
                batch_chars = 0
            batch.append(chunk)
            batch_chars += chunk_chars
            chunks_seen += 1
            original_chars += int(chunk["original_chars"])
            truncated_chunks += int(bool(chunk["truncated"]))
        if batch:
            batch_no += 1
            res = flush_batch(batch, batch_no=batch_no, out_dir=run_dir, dry_run_script=args.dry_run_script, timeout=args.command_timeout)
            commands.append(res)
            if res["command"]["returncode"] != 0 or res["command"].get("timed_out"):
                blockers.append(f"gliner_batch_failed:{batch_no}")
    total_claims = sum(int(c.get("claim_packet_count") or 0) for c in commands)
    receipt = {
        "schema": "lucidota.chroma_gliner_bounded_probe.v1",
        "generated_at": now_z(),
        "sqlite_path": rel(db),
        "sqlite_open_mode": "read_only",
        "total_chunks_reported_by_embeddings": total_chunks,
        "limit": args.limit,
        "start_after_id": args.start_after_id,
        "page_size": args.page_size,
        "chunks_seen": chunks_seen,
        "original_chars_seen": original_chars,
        "truncated_chunks": truncated_chunks,
        "max_chars_per_chunk": args.max_chars_per_chunk,
        "max_chars_per_run": args.max_chars_per_run,
        "batch_count": batch_no,
        "total_claim_packets": total_claims,
        "commands": commands,
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "canonical_graph_mutation": False,
        "blockers": blockers,
        "status": "PASS" if not blockers and chunks_seen == args.limit else "FAIL",
    }
    receipt_path = run_dir / "chroma_gliner_bounded_probe_receipt.json"
    receipt["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(receipt_path)}")
    print(f"CHROMA_CHUNKS_STREAMED={chunks_seen}")
    print(f"GLINER_BATCHES={batch_no}")
    print(f"CHROMA_GLINER_PROBE={receipt['status']}")
    return 0 if receipt["status"] == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
