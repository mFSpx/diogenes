#!/usr/bin/env python3
"""Validate Chrono replay cursor rows and pending replay target counts."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/chrono_ledger"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"chrono_replay_cursor_validator_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def validate(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('lucidota_korpus.chrono_replay_cursor') AS reg")
            if cur.fetchone()["reg"] is None:
                blockers.append("chrono_replay_cursor_missing")
                rows = []
                file_count = 0
                max_boundary = None
            else:
                cur.execute("SELECT count(*) AS n FROM lucidota_korpus.file_object")
                file_count = int(cur.fetchone()["n"])
                cur.execute("SELECT file_uuid::text, first_seen_at::text FROM lucidota_korpus.file_object ORDER BY first_seen_at DESC, file_uuid DESC LIMIT 1")
                max_boundary = dict(cur.fetchone() or {})
                cur.execute(
                    """
                    SELECT cursor_name, last_file_uuid::text, last_file_first_seen_at::text,
                           processed_count, last_replay_started_at::text, last_replay_finished_at::text, updated_at::text
                    FROM lucidota_korpus.chrono_replay_cursor
                    ORDER BY cursor_name
                    """
                )
                cursor_rows = [dict(r) for r in cur.fetchall()]
                rows = []
                for row in cursor_rows:
                    pending = None
                    last_exists = True
                    beyond_max = False
                    if row["last_file_uuid"]:
                        cur.execute("SELECT count(*) AS n FROM lucidota_korpus.file_object WHERE file_uuid=%s::uuid", (row["last_file_uuid"],))
                        last_exists = int(cur.fetchone()["n"]) == 1
                    if row["last_file_first_seen_at"] and max_boundary:
                        cur.execute(
                            """
                            SELECT (%s::timestamptz, %s::uuid) > (%s::timestamptz, %s::uuid) AS beyond
                            """,
                            (row["last_file_first_seen_at"], row["last_file_uuid"] or "00000000-0000-0000-0000-000000000000", max_boundary["first_seen_at"], max_boundary["file_uuid"]),
                        )
                        beyond_max = bool(cur.fetchone()["beyond"])
                    cur.execute(
                        """
                        SELECT count(*) AS n
                        FROM lucidota_korpus.file_object f
                        WHERE %s::timestamptz IS NULL
                           OR (f.first_seen_at, f.file_uuid) > (%s::timestamptz, %s::uuid)
                        """,
                        (row["last_file_first_seen_at"], row["last_file_first_seen_at"], row["last_file_uuid"] or "00000000-0000-0000-0000-000000000000"),
                    )
                    pending = int(cur.fetchone()["n"])
                    item = {**row, "last_file_uuid_exists": last_exists, "boundary_beyond_max_file": beyond_max, "pending_target_count": pending}
                    rows.append(item)
                    if not last_exists:
                        blockers.append(f"last_file_uuid_missing:{row['cursor_name']}")
                    if beyond_max:
                        blockers.append(f"cursor_boundary_beyond_max_file:{row['cursor_name']}")
                    if row["processed_count"] < 0:
                        blockers.append(f"processed_count_negative:{row['cursor_name']}")
    if not rows and not blockers:
        blockers.append("no_cursor_rows")
    report = {"action": "validate", "status": "PASS" if not blockers else "FAIL", "execute_performed": False, "db_writes_performed": False, "graph_writes_performed": False, "file_object_count": file_count, "max_file_boundary": max_boundary, "cursors": rows, "blockers": blockers}
    write_report("pass" if not blockers else "fail", report)
    print("CHRONO_CURSOR_VALIDATION=" + report["status"])
    print(f"CURSOR_ROWS={len(rows)}")
    print(f"TOTAL_PENDING_TARGETS={sum(int(r.get('pending_target_count') or 0) for r in rows)}")
    return 0 if not blockers else 5


def main() -> int:
    p = argparse.ArgumentParser(description="Validate Chrono replay cursors and pending targets")
    p.add_argument("--database-url")
    p.set_defaults(func=validate)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
