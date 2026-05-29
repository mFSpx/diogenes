#!/usr/bin/env python3
"""Krampus watcher: enqueue new files from KRAMPUSCHEWING into ABSURD.

Watches /KRAMPUSCHEWING/ for new files, deduplicates by SHA256, and enqueues
jobs to lucidota_control.absurd_queue_job (queue_name='krampus_ingest').
Writes detection receipts to 05_OUTPUTS/krampus_ingest/<uuid>/detected.json.

Modes:
  --once      scan existing files not yet queued, enqueue them, exit
  --watch     run continuous watchdog (or polling fallback) loop
  --dry-run   print what would be queued, no DB writes, no receipts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
WATCH_DIR = ROOT / "KRAMPUSCHEWING"
OUT_BASE = ROOT / "05_OUTPUTS" / "krampus_ingest"
QUEUE_NAME = "krampus_ingest"
JOB_KIND = "intake"
WORKFLOW_NAME = "krampus-ingest"
POLL_INTERVAL_SEC = 10

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("krampus_watcher")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def db_url() -> str:
    return (
        os.environ.get("LUCIDOTA_GO_STATE_DSN")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def write_receipt(job_uuid: str, payload: dict[str, Any]) -> Path:
    out_dir = OUT_BASE / job_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "detected.json"
    receipt: dict[str, Any] = {
        "schema": "lucidota.krampus_watcher.detected.v1",
        "generated_at": now_iso(),
        "job_uuid": job_uuid,
        "queue_name": QUEUE_NAME,
        "job_kind": JOB_KIND,
        "payload": payload,
        "receipt_path": rel(receipt_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=False), encoding="utf-8")
    log.info("receipt written: %s", rel(receipt_path))
    return receipt_path


def sha256_already_queued(cur: Any, sha256: str) -> bool:
    """Return True if a job with this SHA256 in payload already exists."""
    cur.execute(
        """
        SELECT 1 FROM lucidota_control.absurd_queue_job
        WHERE queue_name = %s
          AND (payload->>'sha256') = %s
        LIMIT 1
        """,
        (QUEUE_NAME, sha256),
    )
    return cur.fetchone() is not None


def enqueue_file(path: Path, *, dry_run: bool = False) -> dict[str, Any] | None:
    """Compute SHA256, check dedup, insert job row, write receipt.

    Returns the job dict on success, None if skipped or dry-run.
    """
    if not path.is_file():
        log.debug("skip non-file: %s", path)
        return None

    try:
        sha256 = sha256_file(path)
    except OSError as exc:
        log.warning("cannot hash %s: %s", path, exc)
        return None

    detected_at = now_iso()
    payload: dict[str, Any] = {
        "file_path": str(path.resolve()),
        "sha256": sha256,
        "detected_at": detected_at,
    }
    job_uuid = str(uuid.uuid4())
    idempotency_key = f"krampus_ingest:{sha256}"

    if dry_run:
        log.info("[dry-run] would enqueue %s sha256=%s", rel(path), sha256[:12])
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "file_path": str(path.resolve()),
                    "sha256": sha256,
                    "job_uuid": job_uuid,
                    "idempotency_key": idempotency_key,
                    "queue_name": QUEUE_NAME,
                    "job_kind": JOB_KIND,
                },
                sort_keys=False,
            )
        )
        return None

    url = db_url()
    try:
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                if sha256_already_queued(cur, sha256):
                    log.debug("already queued sha256=%s path=%s", sha256[:12], rel(path))
                    return None
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_job
                      (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, payload)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (queue_name, idempotency_key) DO UPDATE
                      SET updated_at = now()
                    RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                    """,
                    (
                        job_uuid,
                        QUEUE_NAME,
                        WORKFLOW_NAME,
                        JOB_KIND,
                        idempotency_key,
                        json.dumps(payload),
                    ),
                )
                row = cur.fetchone()
                returned_uuid = str(row[0])
                inserted_new = bool(row[1])
                if inserted_new:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.absurd_queue_event
                          (job_uuid, queue_name, event_kind, event_source, detail)
                        VALUES (%s::uuid, %s, 'enqueued', 'krampus_watcher', %s::jsonb)
                        """,
                        (
                            returned_uuid,
                            QUEUE_NAME,
                            json.dumps(
                                {
                                    "file_path": str(path.resolve()),
                                    "sha256": sha256,
                                    "detected_at": detected_at,
                                }
                            ),
                        ),
                    )
            conn.commit()
    except Exception as exc:
        log.error("DB error enqueuing %s: %s", rel(path), exc)
        return None

    job: dict[str, Any] = {
        "job_uuid": returned_uuid,
        "inserted_new": inserted_new,
        "queue_name": QUEUE_NAME,
        "job_kind": JOB_KIND,
        "idempotency_key": idempotency_key,
        "payload": payload,
    }
    write_receipt(returned_uuid, payload)
    log.info(
        "enqueued job_uuid=%s inserted_new=%s sha256=%s path=%s",
        returned_uuid,
        inserted_new,
        sha256[:12],
        rel(path),
    )
    return job


def scan_existing(watch_dir: Path, *, dry_run: bool = False) -> list[dict[str, Any]]:
    """Walk watch_dir recursively, enqueue files not already queued."""
    results: list[dict[str, Any]] = []
    for p in sorted(watch_dir.rglob("*")):
        if not p.is_file():
            continue
        job = enqueue_file(p, dry_run=dry_run)
        if job:
            results.append(job)
    return results


# ---------------------------------------------------------------------------
# Watchdog-backed watch loop (preferred)
# ---------------------------------------------------------------------------

def _make_watchdog_handler(dry_run: bool) -> Any:
    from watchdog.events import FileSystemEventHandler  # type: ignore[import]

    class _Handler(FileSystemEventHandler):
        def on_created(self, event: Any) -> None:
            if event.is_directory:
                return
            path = Path(str(event.src_path))
            # Brief settle delay so the file is fully written
            time.sleep(0.25)
            enqueue_file(path, dry_run=dry_run)

        def on_moved(self, event: Any) -> None:
            if event.is_directory:
                return
            path = Path(str(event.dest_path))
            time.sleep(0.25)
            enqueue_file(path, dry_run=dry_run)

    return _Handler()


def watch_watchdog(watch_dir: Path, *, dry_run: bool = False) -> None:
    from watchdog.observers import Observer  # type: ignore[import]

    handler = _make_watchdog_handler(dry_run)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()
    log.info("watchdog observer started on %s", watch_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping observer")
    finally:
        observer.stop()
        observer.join()


# ---------------------------------------------------------------------------
# Polling fallback
# ---------------------------------------------------------------------------

def watch_polling(watch_dir: Path, *, dry_run: bool = False, interval: int = POLL_INTERVAL_SEC) -> None:
    log.info("watchdog not available — using polling fallback (interval=%ss)", interval)
    seen: set[str] = set()
    # seed with already-queued files so we do not re-enqueue on startup
    for p in watch_dir.rglob("*"):
        if p.is_file():
            try:
                seen.add(sha256_file(p))
            except OSError:
                pass
    log.info("polling seed: %d files already seen", len(seen))
    try:
        while True:
            for p in sorted(watch_dir.rglob("*")):
                if not p.is_file():
                    continue
                try:
                    sha256 = sha256_file(p)
                except OSError:
                    continue
                if sha256 in seen:
                    continue
                seen.add(sha256)
                enqueue_file(p, dry_run=dry_run)
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — polling stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Krampus watcher: enqueue new files from KRAMPUSCHEWING into ABSURD "
            "(queue_name=krampus_ingest, job_kind=intake)."
        )
    )
    ap.add_argument(
        "--watch-dir",
        default=str(WATCH_DIR),
        help="Directory to watch (default: LUCIDOTA/KRAMPUSCHEWING)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be queued; no DB writes, no receipts.",
    )

    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Scan existing files not yet queued, enqueue them, then exit.",
    )
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="Run continuous file-watch loop (watchdog or polling fallback).",
    )

    ap.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL_SEC,
        help=f"Polling interval in seconds for fallback mode (default: {POLL_INTERVAL_SEC}).",
    )
    ap.add_argument(
        "--database-url",
        default=None,
        help="Override DB URL (default: LUCIDOTA_GO_STATE_DSN or postgresql:///lucidota_state).",
    )

    args = ap.parse_args()

    # Allow CLI override of DB URL
    if args.database_url:
        os.environ["LUCIDOTA_GO_STATE_DSN"] = args.database_url

    watch_dir = Path(args.watch_dir)
    if not watch_dir.exists():
        log.error("watch_dir does not exist: %s", watch_dir)
        return 1

    if args.once:
        jobs = scan_existing(watch_dir, dry_run=args.dry_run)
        log.info("--once complete: %d jobs enqueued", len(jobs))
        return 0

    # --watch
    try:
        import watchdog  # noqa: F401 — probe import only
        watch_watchdog(watch_dir, dry_run=args.dry_run)
    except ImportError:
        watch_polling(watch_dir, dry_run=args.dry_run, interval=args.poll_interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
