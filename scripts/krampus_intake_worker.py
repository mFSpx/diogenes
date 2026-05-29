#!/usr/bin/env python3
"""ABSURD worker #1 for krampus_ingest: intake job handler.

Dequeues intake jobs from queue_name='krampus_ingest', validates the file,
computes file metadata, writes an intake receipt, and enqueues a route_extract
follow-on job. Never mutates canonical graph tables.

mutation_class: queue_writer (receipt_only for 05_OUTPUTS writes)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "05_OUTPUTS" / "krampus_ingest"

QUEUE_NAME = "krampus_ingest"
WORKER_KEY = "krampus_intake_v1"
JOB_KIND_INTAKE = "intake"
JOB_KIND_NEXT = "route_extract"
WORKFLOW_NAME = "krampus-ingest-pipeline"

POLL_SLEEP_SEC = 5
MAX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def db_url() -> str:
    return (
        os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("LUCIDOTA_GO_STATE_DSN")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def connect() -> psycopg.Connection:
    return psycopg.connect(db_url())


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

# Magic-number signatures: (offset, bytes) -> (mime, is_text, is_image, is_archive)
_MAGIC_SIGS: list[tuple[int, bytes, str, bool, bool, bool]] = [
    # offset, header, mime, is_text, is_image, is_archive
    (0, b"\xff\xd8\xff",                 "image/jpeg",                 False, True,  False),
    (0, b"\x89PNG\r\n\x1a\n",           "image/png",                  False, True,  False),
    (0, b"GIF87a",                       "image/gif",                  False, True,  False),
    (0, b"GIF89a",                       "image/gif",                  False, True,  False),
    (0, b"BM",                           "image/bmp",                  False, True,  False),
    (0, b"RIFF",                         "image/webp",                 False, True,  False),
    (0, b"\x00\x00\x01\x00",            "image/x-ico",                False, True,  False),
    (0, b"II*\x00",                      "image/tiff",                 False, True,  False),
    (0, b"MM\x00*",                      "image/tiff",                 False, True,  False),
    (0, b"PK\x03\x04",                   "application/zip",            False, False, True),
    (0, b"PK\x05\x06",                   "application/zip",            False, False, True),
    (0, b"\x1f\x8b",                     "application/gzip",           False, False, True),
    (0, b"BZh",                          "application/x-bzip2",        False, False, True),
    (0, b"\xfd7zXZ\x00",                "application/x-xz",           False, False, True),
    (0, b"Rar!\x1a\x07",                "application/x-rar",          False, False, True),
    (0, b"7z\xbc\xaf\x27\x1c",         "application/x-7z-compressed", False, False, True),
    (0, b"%PDF",                         "application/pdf",            False, False, False),
    (0, b"\xd0\xcf\x11\xe0",            "application/msoffice",       False, False, False),
    (0, b"<!DOCTYPE",                    "text/html",                  True,  False, False),
    (0, b"<html",                        "text/html",                  True,  False, False),
    (0, b"<HTML",                        "text/html",                  True,  False, False),
    (0, b"<?xml",                        "text/xml",                   True,  False, False),
    (0, b"{",                            "application/json",           True,  False, False),
    (0, b"[",                            "application/json",           True,  False, False),
]

_EXT_MAP: dict[str, tuple[str, bool, bool, bool]] = {
    # ext -> (mime, is_text, is_image, is_archive)
    ".txt":   ("text/plain",              True,  False, False),
    ".md":    ("text/markdown",           True,  False, False),
    ".rst":   ("text/x-rst",              True,  False, False),
    ".py":    ("text/x-python",           True,  False, False),
    ".js":    ("text/javascript",         True,  False, False),
    ".ts":    ("text/typescript",         True,  False, False),
    ".json":  ("application/json",        True,  False, False),
    ".yaml":  ("text/yaml",               True,  False, False),
    ".yml":   ("text/yaml",               True,  False, False),
    ".toml":  ("text/toml",               True,  False, False),
    ".csv":   ("text/csv",                True,  False, False),
    ".tsv":   ("text/tab-separated-values", True, False, False),
    ".html":  ("text/html",               True,  False, False),
    ".htm":   ("text/html",               True,  False, False),
    ".xml":   ("text/xml",                True,  False, False),
    ".sql":   ("text/x-sql",              True,  False, False),
    ".sh":    ("text/x-shellscript",      True,  False, False),
    ".rs":    ("text/x-rust",             True,  False, False),
    ".go":    ("text/x-go",               True,  False, False),
    ".c":     ("text/x-c",                True,  False, False),
    ".h":     ("text/x-c",                True,  False, False),
    ".cpp":   ("text/x-c++",              True,  False, False),
    ".java":  ("text/x-java",             True,  False, False),
    ".log":   ("text/plain",              True,  False, False),
    ".png":   ("image/png",               False, True,  False),
    ".jpg":   ("image/jpeg",              False, True,  False),
    ".jpeg":  ("image/jpeg",              False, True,  False),
    ".gif":   ("image/gif",               False, True,  False),
    ".bmp":   ("image/bmp",               False, True,  False),
    ".webp":  ("image/webp",              False, True,  False),
    ".tiff":  ("image/tiff",              False, True,  False),
    ".tif":   ("image/tiff",              False, True,  False),
    ".ico":   ("image/x-ico",             False, True,  False),
    ".svg":   ("image/svg+xml",           True,  True,  False),
    ".pdf":   ("application/pdf",         False, False, False),
    ".zip":   ("application/zip",         False, False, True),
    ".gz":    ("application/gzip",        False, False, True),
    ".bz2":   ("application/x-bzip2",     False, False, True),
    ".xz":    ("application/x-xz",        False, False, True),
    ".tar":   ("application/x-tar",       False, False, True),
    ".rar":   ("application/x-rar",       False, False, True),
    ".7z":    ("application/x-7z-compressed", False, False, True),
}


def _try_magic(path: Path) -> tuple[str, bool, bool, bool] | None:
    """Attempt to use python-magic. Returns None if unavailable."""
    try:
        import magic  # type: ignore[import]
        mime: str = magic.from_file(str(path), mime=True)
        is_text = mime.startswith("text/")
        is_image = mime.startswith("image/")
        is_archive = mime in {
            "application/zip",
            "application/gzip",
            "application/x-gzip",
            "application/x-bzip2",
            "application/x-xz",
            "application/x-rar",
            "application/x-7z-compressed",
            "application/x-tar",
            "application/x-compress",
        }
        return mime, is_text, is_image, is_archive
    except Exception:
        return None


def _detect_from_header(path: Path) -> tuple[str, bool, bool, bool] | None:
    try:
        with open(path, "rb") as fh:
            header = fh.read(256)
    except OSError:
        return None
    for offset, sig, mime, is_text, is_image, is_archive in _MAGIC_SIGS:
        chunk = header[offset: offset + len(sig)]
        if chunk == sig:
            return mime, is_text, is_image, is_archive
    return None


def _detect_is_text_from_bytes(header: bytes) -> bool:
    """Heuristic: if no null bytes in first 512 bytes, treat as text."""
    return b"\x00" not in header


def identify_file(path: Path) -> dict[str, Any]:
    magic_result = _try_magic(path)
    if magic_result:
        mime, is_text, is_image, is_archive = magic_result
        method = "python_magic"
    else:
        header_result = _detect_from_header(path)
        if header_result:
            mime, is_text, is_image, is_archive = header_result
            method = "header_bytes"
        else:
            # Fall back to extension
            ext = path.suffix.lower()
            if ext in _EXT_MAP:
                mime, is_text, is_image, is_archive = _EXT_MAP[ext]
                method = "extension"
            else:
                # Last resort: sniff for text
                try:
                    with open(path, "rb") as fh:
                        snippet = fh.read(512)
                    is_text = _detect_is_text_from_bytes(snippet)
                except OSError:
                    is_text = False
                mime = "text/plain" if is_text else "application/octet-stream"
                is_image = False
                is_archive = False
                method = "heuristic"
    return {
        "mime_type": mime,
        "is_text": is_text,
        "is_image": is_image,
        "is_archive": is_archive,
        "detection_method": method,
    }


# ---------------------------------------------------------------------------
# Metadata computation
# ---------------------------------------------------------------------------

def compute_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    file_size_bytes = stat.st_size
    type_info = identify_file(path)
    line_count: int | None = None
    if type_info["is_text"]:
        try:
            with open(path, "rb") as fh:
                line_count = sum(1 for _ in fh)
        except OSError:
            line_count = None
    token_count_norm = min(file_size_bytes / 100_000.0, 1.0)
    return {
        "file_size_bytes": file_size_bytes,
        "mime_type": type_info["mime_type"],
        "is_text": type_info["is_text"],
        "is_image": type_info["is_image"],
        "is_archive": type_info["is_archive"],
        "detection_method": type_info["detection_method"],
        "line_count": line_count,
        "token_count_norm": round(token_count_norm, 6),
    }


# ---------------------------------------------------------------------------
# Receipt writer
# ---------------------------------------------------------------------------

def write_receipt(job_uuid: str, payload: dict[str, Any], metadata: dict[str, Any], next_job_uuid: str | None) -> Path:
    job_dir = OUT_BASE / job_uuid
    job_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "lucidota.krampus_intake.receipt.v1",
        "receipt_mode": "ABSURD_POSTGRES_RUNTIME",
        "generated_at": now_iso(),
        "job_uuid": job_uuid,
        "queue_name": QUEUE_NAME,
        "job_kind": JOB_KIND_INTAKE,
        "worker_key": WORKER_KEY,
        "file_path": payload.get("file_path"),
        "sha256": payload.get("sha256"),
        "next_job_uuid": next_job_uuid,
        "metadata": metadata,
    }
    out = job_dir / "intake.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"RECEIPT_PATH={out.relative_to(ROOT)}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

def ensure_queue_exists(cur: psycopg.Cursor) -> None:
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue
          (queue_name, owner_subsystem, max_attempts, notes)
        VALUES (%s, 'KRAMPUSCHEWING intake pipeline', %s, 'ABSURD krampus ingest pipeline queue')
        ON CONFLICT (queue_name) DO NOTHING
        """,
        (QUEUE_NAME, MAX_ATTEMPTS),
    )


def enqueue_next(
    cur: psycopg.Cursor,
    *,
    file_path: str,
    sha256: str,
    metadata: dict[str, Any],
    source_job_uuid: str,
) -> str:
    payload = {
        "file_path": file_path,
        "sha256": sha256,
        "source_job_uuid": source_job_uuid,
        **metadata,
    }
    idem = sha256_obj({"queue": QUEUE_NAME, "job_kind": JOB_KIND_NEXT, "file_path": file_path, "sha256": sha256})
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_job
          (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts)
        VALUES (%s, %s, %s, %s, %s::jsonb, 100, %s)
        ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
        RETURNING job_uuid::text
        """,
        (
            QUEUE_NAME,
            WORKFLOW_NAME,
            JOB_KIND_NEXT,
            idem,
            json.dumps(payload, default=str),
            MAX_ATTEMPTS,
        ),
    )
    row = cur.fetchone()
    next_uuid = row[0] if row else ""
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_event
          (job_uuid, queue_name, event_kind, event_source, detail)
        VALUES (%s::uuid, %s, 'enqueued', %s, %s::jsonb)
        """,
        (
            next_uuid,
            QUEUE_NAME,
            WORKER_KEY,
            json.dumps({"source_job_uuid": source_job_uuid, "job_kind": JOB_KIND_NEXT}),
        ),
    )
    return next_uuid


# ---------------------------------------------------------------------------
# Core: process one job
# ---------------------------------------------------------------------------

def process_one(*, dry_run: bool = False) -> bool:
    """Claim and process one intake job. Returns True if a job was found and processed."""
    worker_id = f"{socket.gethostname()}:{os.getpid()}"

    with connect() as conn:
        with conn.cursor() as cur:
            ensure_queue_exists(cur)

            if dry_run:
                cur.execute(
                    """
                    SELECT job_uuid::text, payload, attempt_count, max_attempts
                    FROM lucidota_control.absurd_queue_job
                    WHERE queue_name=%s
                      AND status='queued'
                      AND job_kind=%s
                      AND run_after<=now()
                    ORDER BY priority, created_at
                    LIMIT 1
                    """,
                    (QUEUE_NAME, JOB_KIND_INTAKE),
                )
                row = cur.fetchone()
                if not row:
                    print(f"[{now_iso()}] dry-run: no queued intake jobs", file=sys.stderr)
                    conn.rollback()
                    return False
                job_uuid, payload, attempt_count, max_attempts = row
                payload = dict(payload or {})
                print(
                    f"[{now_iso()}] dry-run: would process job_uuid={job_uuid} "
                    f"file_path={payload.get('file_path')} attempt={attempt_count}/{max_attempts}",
                    file=sys.stderr,
                )
                conn.rollback()
                return True

            # --- execute path ---
            cur.execute(
                """
                SELECT job_uuid::text, payload, attempt_count, max_attempts, idempotency_key, workflow_name
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s
                  AND status='queued'
                  AND job_kind=%s
                  AND run_after<=now()
                ORDER BY priority, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (QUEUE_NAME, JOB_KIND_INTAKE),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return False

            job_uuid, payload, attempt_count, max_attempts, idempotency_key, workflow_name = row
            payload = dict(payload or {})
            attempt_count = int(attempt_count)
            max_attempts = int(max_attempts)

            print(
                f"[{now_iso()}] claimed job_uuid={job_uuid} attempt={attempt_count+1}/{max_attempts}",
                file=sys.stderr,
            )

            # Mark running
            cur.execute(
                """
                UPDATE lucidota_control.absurd_queue_job
                SET status='running',
                    locked_by=%s,
                    locked_at=now(),
                    leased_by=%s,
                    lease_expires_at=now()+interval '5 minutes',
                    last_heartbeat_at=now(),
                    attempt_count=attempt_count+1
                WHERE job_uuid=%s::uuid
                """,
                (worker_id, worker_id, job_uuid),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event
                  (job_uuid, queue_name, event_kind, event_source, detail)
                VALUES (%s::uuid, %s, 'started', %s, %s::jsonb)
                """,
                (job_uuid, QUEUE_NAME, WORKER_KEY, json.dumps({"worker_id": worker_id})),
            )

            try:
                # Validate payload
                file_path_str = payload.get("file_path")
                if not file_path_str:
                    raise ValueError("payload missing file_path")
                file_path = Path(file_path_str)
                if not file_path.exists():
                    raise FileNotFoundError(f"file_path does not exist: {file_path_str}")
                if not file_path.is_file():
                    raise ValueError(f"file_path is not a regular file: {file_path_str}")

                # Compute metadata
                metadata = compute_metadata(file_path)
                file_sha256 = sha256_file(file_path)

                # Enqueue next job
                next_job_uuid = enqueue_next(
                    cur,
                    file_path=str(file_path),
                    sha256=file_sha256,
                    metadata=metadata,
                    source_job_uuid=job_uuid,
                )

                # Mark succeeded
                result = {
                    "ok": True,
                    "outcome": "succeeded",
                    "file_path": str(file_path),
                    "sha256": file_sha256,
                    "next_job_uuid": next_job_uuid,
                    **metadata,
                }
                cur.execute(
                    """
                    UPDATE lucidota_control.absurd_queue_job
                    SET status='succeeded',
                        result=%s::jsonb,
                        completed_at=now(),
                        updated_at=now(),
                        last_error='',
                        error_kind='',
                        error_message=''
                    WHERE job_uuid=%s::uuid
                    """,
                    (json.dumps(result, default=str), job_uuid),
                )
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_event
                      (job_uuid, queue_name, event_kind, event_source, detail)
                    VALUES (%s::uuid, %s, 'succeeded', %s, %s::jsonb)
                    """,
                    (job_uuid, QUEUE_NAME, WORKER_KEY, json.dumps(result, default=str)),
                )

                conn.commit()

                # Write receipt after commit (side effect only, not rolled back)
                write_receipt(job_uuid, {**payload, "sha256": file_sha256}, metadata, next_job_uuid)
                print(
                    f"[{now_iso()}] succeeded job_uuid={job_uuid} next={next_job_uuid}",
                    file=sys.stderr,
                )

            except Exception as exc:
                conn.rollback()
                error_kind = type(exc).__name__
                error_message = str(exc)
                final_attempt = (attempt_count + 1) >= max_attempts
                new_status = "dead_lettered" if final_attempt else "failed"

                print(
                    f"[{now_iso()}] {new_status} job_uuid={job_uuid} error={error_kind}: {error_message}",
                    file=sys.stderr,
                )

                with connect() as conn2:
                    with conn2.cursor() as cur2:
                        result_err = {
                            "ok": False,
                            "outcome": new_status,
                            "error_kind": error_kind,
                            "error_message": error_message,
                        }
                        cur2.execute(
                            """
                            UPDATE lucidota_control.absurd_queue_job
                            SET status=%s,
                                result=%s::jsonb,
                                updated_at=now(),
                                last_error=%s,
                                error_kind=%s,
                                error_message=%s
                            WHERE job_uuid=%s::uuid
                            """,
                            (
                                new_status,
                                json.dumps(result_err, default=str),
                                error_message,
                                error_kind,
                                error_message,
                                job_uuid,
                            ),
                        )
                        cur2.execute(
                            """
                            INSERT INTO lucidota_control.absurd_queue_event
                              (job_uuid, queue_name, event_kind, event_source, detail)
                            VALUES (%s::uuid, %s, %s, %s, %s::jsonb)
                            """,
                            (
                                job_uuid,
                                QUEUE_NAME,
                                new_status if new_status in ("failed", "dead_lettered") else "failed",
                                WORKER_KEY,
                                json.dumps(result_err, default=str),
                            ),
                        )
                        if final_attempt:
                            cur2.execute(
                                """
                                INSERT INTO lucidota_control.absurd_queue_dead_letter
                                  (job_uuid, queue_name, workflow_name, job_kind, idempotency_key,
                                   error_kind, error_message, attempt_count, payload_sha256, context)
                                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                                ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET
                                  error_kind=EXCLUDED.error_kind,
                                  error_message=EXCLUDED.error_message,
                                  attempt_count=EXCLUDED.attempt_count,
                                  last_seen_at=now(),
                                  context=EXCLUDED.context
                                """,
                                (
                                    job_uuid,
                                    QUEUE_NAME,
                                    workflow_name,
                                    JOB_KIND_INTAKE,
                                    idempotency_key,
                                    error_kind,
                                    error_message,
                                    attempt_count + 1,
                                    sha256_obj(payload),
                                    json.dumps(result_err, default=str),
                                ),
                            )
                        conn2.commit()

                return True  # job was found (though it failed)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ABSURD krampus_ingest intake worker (job_kind='intake')."
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Process exactly one job then exit.",
    )
    ap.add_argument(
        "--loop",
        action="store_true",
        help=f"Poll continuously with {POLL_SLEEP_SEC}s sleep between empty polls.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without any writes.",
    )
    args = ap.parse_args()

    if not args.once and not args.loop:
        ap.print_help(sys.stderr)
        return 1

    if args.once:
        found = process_one(dry_run=args.dry_run)
        if not found:
            print(f"[{now_iso()}] no queued intake jobs", file=sys.stderr)
        return 0

    # --loop
    print(f"[{now_iso()}] starting loop (dry_run={args.dry_run})", file=sys.stderr)
    while True:
        try:
            found = process_one(dry_run=args.dry_run)
            if not found:
                time.sleep(POLL_SLEEP_SEC)
        except KeyboardInterrupt:
            print(f"[{now_iso()}] interrupted", file=sys.stderr)
            return 0
        except Exception as exc:
            print(f"[{now_iso()}] unhandled exception: {exc}", file=sys.stderr)
            time.sleep(POLL_SLEEP_SEC)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
