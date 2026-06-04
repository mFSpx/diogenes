#!/usr/bin/env python3
"""Archive heavy CLI receipt payload tails to compressed cold storage."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "payload_archive"
DB_DSN = os.environ.get("LUCIDOTA_STATE_DSN") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_output_dir() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def write_archive(path: Path, payload: dict[str, object]) -> tuple[str, int, int]:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(raw)
    return sha256_text(raw.decode("utf-8")), len(raw), len(payload.get("payload_text", "") or "")


def select_candidates(conn, *, older_than_hours: int, max_rows: int, archive_all: bool):
    where_clause = ""
    params = [max_rows]
    if not archive_all:
        where_clause = "AND received_at < now() - (%s || ' hours')::interval"
        params = [older_than_hours, max_rows]
    query = f"""
        SELECT
            receipt_uuid,
            received_at,
            command_line,
            status,
            stdout_tail,
            stderr_tail,
            stdout_tail_sha256,
            stderr_tail_sha256,
            stdout_archive_ref,
            stderr_archive_ref,
            stdout_archived_at,
            stderr_archived_at,
            receipt_path,
            timeout_seconds,
            restart_count
        FROM lucidota_control.cli_process_receipt
        WHERE (coalesce(stdout_tail, '') <> '' OR coalesce(stderr_tail, '') <> '')
          AND (coalesce(stdout_archive_ref, '') = '' OR coalesce(stderr_archive_ref, '') = '')
          {where_clause}
        ORDER BY received_at ASC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def archive_one(conn, row: dict[str, object]) -> list[dict[str, object]]:
    archived: list[dict[str, object]] = []
    receipt_uuid = str(row["receipt_uuid"])
    base_fields = {
        "source_table": "lucidota_control.cli_process_receipt",
        "source_uuid": receipt_uuid,
        "receipt_uuid": receipt_uuid,
        "received_at": row["received_at"].isoformat() if hasattr(row["received_at"], "isoformat") else str(row["received_at"]),
        "command_line": row.get("command_line", ""),
        "status": row.get("status", ""),
        "receipt_path": row.get("receipt_path", ""),
        "timeout_seconds": row.get("timeout_seconds", 0),
        "restart_count": row.get("restart_count", 0),
    }
    for kind, field in (("stdout_tail", "stdout_tail"), ("stderr_tail", "stderr_tail")):
        payload_text = str(row.get(field) or "")
        if not payload_text:
            continue
        payload = {**base_fields, "payload_kind": kind, "payload_text": payload_text}
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        payload_hash = sha256_text(payload_json)
        archive_path = OUT / "cli_process_receipt" / receipt_uuid / f"{kind}_{payload_hash[:16]}.json.gz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive_path, "wb") as fh:
            fh.write(payload_json.encode("utf-8"))
        archive_ref = rel(archive_path)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.payload_archive (
                    source_table, source_uuid, payload_kind, payload_hash, payload_bytes, payload_chars, archive_path, detail
                ) VALUES (
                    %s, %s::uuid, %s, %s, %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (source_table, source_uuid, payload_kind) DO UPDATE SET
                    payload_hash = EXCLUDED.payload_hash,
                    payload_bytes = EXCLUDED.payload_bytes,
                    payload_chars = EXCLUDED.payload_chars,
                    archive_path = EXCLUDED.archive_path,
                    archived_at = now(),
                    detail = EXCLUDED.detail
                """,
                (
                    "lucidota_control.cli_process_receipt",
                    receipt_uuid,
                    kind,
                    payload_hash,
                    len(payload_json.encode("utf-8")),
                    len(payload_text),
                    archive_ref,
                    json.dumps({"command_line": base_fields["command_line"], "status": base_fields["status"]}, sort_keys=True),
                ),
            )
            cur.execute(
                f"""
                UPDATE lucidota_control.cli_process_receipt
                SET {field} = '',
                    {field}_sha256 = %s,
                    {field.replace('_tail', '_archive_ref')} = %s,
                    {field.replace('_tail', '_archived_at')} = now(),
                    updated_at = now()
                WHERE receipt_uuid = %s::uuid
                """,
                (payload_hash, archive_ref, receipt_uuid),
            )
        archived.append(
            {
                "receipt_uuid": receipt_uuid,
                "payload_kind": kind,
                "archive_ref": archive_ref,
                "payload_hash": payload_hash,
                "payload_chars": len(payload_text),
            }
        )
    return archived


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive CLI receipt tails to cold storage.")
    ap.add_argument("--older-than-hours", type=int, default=24)
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument("--archive-all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    ensure_output_dir()
    archived: list[dict[str, object]] = []
    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            rows = select_candidates(conn, older_than_hours=args.older_than_hours, max_rows=args.max_rows, archive_all=args.archive_all)
            for row in rows:
                if args.dry_run:
                    archived.append({"receipt_uuid": str(row["receipt_uuid"]), "dry_run": True})
                    continue
                archived.extend(archive_one(conn, row))
        conn.commit()
    report = {
        "schema": "lucidota.cli_payload_retention.v1",
        "archive_dir": rel(OUT),
        "count": len(archived),
        "dry_run": args.dry_run,
        "rows": archived,
        "generated_at": utc_now(),
    }
    if args.json:
        print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
