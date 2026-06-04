#!/usr/bin/env python3
"""No-delete filesystem triage into KRAMPUSCHEWING quarantine plus ABSURD reingest queueing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.krampus_extension_policy import classify_path  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "krampuschewing" / "quarantine"
DEFAULT_QUARANTINE_ROOT = ROOT / "KRAMPUSCHEWING" / "quarantine"
DEFAULT_QUEUE = "krampus_quarantine_reingest"
DEFAULT_WORKFLOW = "krampus_quarantine_reingest"
PY = sys.executable or "python3"

PROTECTED_PATH_PARTS = {
    ".git",
    ".venv",
    "03_VAULT",
    "GOALS",
    "04_RUNTIME",
    "05_OUTPUTS",
    "DB",
    "KRAMPUSCHEWING",
}
PROTECTED_LEAF_NAMES = {"CURRENT_HANDOFF.md", "GOAL_LOG.md", "GOAL_HANDOFF_PROMPT.md"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_protected(path: Path, source_root: Path) -> bool:
    try:
        rel_path = path.resolve().relative_to(source_root.resolve())
    except Exception:
        rel_path = path
    parts = set(rel_path.parts)
    if parts & PROTECTED_PATH_PARTS:
        return True
    return path.name in PROTECTED_LEAF_NAMES


def reason_class_for(path: Path, classification: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    kind = str(classification.get("kind") or "unknown")
    lane = str(classification.get("lane") or "")
    if kind == "archive" or lane == "quarantine_archive":
        return "ARCHIVE_EXTRACT_OUTPUT"
    if kind == "database" or lane == "quarantine_database":
        return "UNREGISTERED_TOOL_CORPSE"
    if kind == "binary" or lane == "quarantine_unknown":
        return "UNREGISTERED_TOOL_CORPSE"
    if any(token in path.name.lower() for token in ("old", "legacy", "snapshot", "duplicate", "copy", "bak", ".bak")):
        return "DUPLICATE_DOC_SNAPSHOT"
    if any(token in path.name.lower() for token in ("manual", "doc", "report")):
        return "GENERATED_MANUAL_OLD_VERSION"
    if any(token in path.name.lower() for token in ("dump", "agent", "assistant", "cli")):
        return "OLD_AGENT_DUMP"
    if any(token in path.suffix.lower() for token in (".tmp", ".cache", ".o", ".so", ".pyc")):
        return "CACHE_OR_BUILD_EXHAUST"
    if kind == "unknown":
        return "UNKNOWN_NEEDS_OPERATOR_LABEL"
    return "BROKEN_PARTIAL_SCRIPT" if kind in {"python", "shell"} else "UNKNOWN_NEEDS_OPERATOR_LABEL"


def ensure_queue_job(conn: Any, *, queue_name: str, workflow_name: str, job_uuid: str, payload: dict[str, Any]) -> tuple[str, bool]:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lucidota_control.absurd_queue_job
              (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
            VALUES (%s::uuid, %s, %s, 'external_command', %s, %s::jsonb, 100, 3, %s::jsonb)
            ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at = now()
            RETURNING job_uuid::text, (xmax = 0) AS inserted_new
            """,
            (
                job_uuid,
                queue_name,
                workflow_name,
                f"krampus_quarantine:{job_uuid}",
                json.dumps(payload, sort_keys=True),
                json.dumps({"source": "krampuschewing_quarantine_triage"}, sort_keys=True),
            ),
        )
        returned_uuid, inserted_new = cur.fetchone()
        if inserted_new:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail)
                VALUES (%s,%s,'enqueued',%s::jsonb)
                """,
                (
                    returned_uuid,
                    queue_name,
                    json.dumps({"workflow": workflow_name, "payload": payload}, sort_keys=True),
                ),
            )
    return str(returned_uuid), bool(inserted_new)


def move_file(source: Path, quarantine_root: Path, move_id: str) -> Path:
    dest = quarantine_root / stamp() / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    return dest


def triage(*, source_root: Path, quarantine_root: Path, execute: bool, reason_class: str | None, max_files: int, database_url: str) -> dict[str, Any]:
    source_root = source_root.resolve()
    quarantine_root = quarantine_root.resolve()
    quarantine_root.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    queued: list[dict[str, Any]] = []

    candidates = [p for p in sorted(source_root.rglob("*")) if p.is_file() and not is_protected(p, source_root)]
    for path in candidates[:max_files]:
        classification = classify_path(path)
        if not classification.get("quarantine") and not reason_class:
            continue
        reason = reason_class_for(path, classification, override=reason_class)
        move_id = str(uuid.uuid4())
        sha256 = sha256_file(path)
        size_bytes = path.stat().st_size
        dest = quarantine_root / stamp() / path.name
        entry = {
            "move_uuid": move_id,
            "source_path": rel(path),
            "dest_path": rel(dest),
            "sha256": sha256,
            "size_bytes": size_bytes,
            "reason_class": reason,
            "operator_delete_performed": False,
            "reingest_status": "queued" if execute else "dry_run",
            "classification": classification,
        }
        records.append(entry)
        if execute:
            moved_dest = move_file(path, quarantine_root, move_id)
            entry["dest_path"] = rel(moved_dest)
            move_root = moved_dest.parent
            payload = {
                "command": [
                    PY,
                    "scripts/krampuschewing_master_index.py",
                    "--root",
                    rel(move_root),
                    "--source-label",
                    "krampus_quarantine",
                ],
                "quarantine_root": rel(move_root),
                "move_uuid": move_id,
                "reason_class": reason,
                "timeout_seconds": 300,
            }
            if database_url:
                with psycopg.connect(database_url) as conn:
                    job_uuid, inserted = ensure_queue_job(
                        conn,
                        queue_name=DEFAULT_QUEUE,
                        workflow_name=DEFAULT_WORKFLOW,
                        job_uuid=move_id,
                        payload=payload,
                    )
                    conn.commit()
                    queued.append({"job_uuid": job_uuid, "inserted_new": inserted, "payload": payload})
                    entry["reingest_status"] = "queued"
                    entry["queue_job_uuid"] = job_uuid
            moved.append(entry)

    report = {
        "schema": "lucidota.krampuschewing.quarantine_move.v1",
        "generated_at": now(),
        "execute_performed": execute,
        "source_root": rel(source_root),
        "quarantine_root": rel(quarantine_root),
        "moved_count": len(moved),
        "queued_count": len(queued),
        "records": records,
        "moved": moved,
        "queued": queued,
        "operator_delete_performed": False,
        "receipt_path": "",
    }
    receipt = OUT / f"krampuschewing_quarantine_move_{stamp()}.json"
    report["receipt_path"] = rel(receipt)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Move quarantine candidates into KRAMPUSCHEWING and queue reingest.")
    ap.add_argument("--source-root", default=str(ROOT))
    ap.add_argument("--quarantine-root", default=str(DEFAULT_QUARANTINE_ROOT))
    ap.add_argument("--reason-class", default=None)
    ap.add_argument("--max-files", type=int, default=2000)
    ap.add_argument("--database-url", default=os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    execute = bool(args.execute)
    report = triage(
        source_root=Path(args.source_root),
        quarantine_root=Path(args.quarantine_root),
        execute=execute,
        reason_class=args.reason_class,
        max_files=args.max_files,
        database_url=args.database_url,
    )
    print("RECEIPT_PATH=" + report["receipt_path"])
    print("KRAMPUS_TRIAGE=" + ("PASS" if report["moved_count"] or not execute else "DRY_RUN"))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
