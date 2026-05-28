#!/usr/bin/env python3
"""Explain Chrono processed_count vs mtime-freeze candidate row delta without mutation."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception as exc:
    raise SystemExit(f"psycopg required: {exc}")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "chrono_ledger"
DB = os.environ.get("DATABASE_URL", "postgresql:///lucidota_storage")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest_mtime = sorted(OUT_DIR.glob("chrono_mtime_freeze_dry_run_*.json"))[-1]
    mtime_report = json.loads(latest_mtime.read_text(encoding="utf-8"))
    with psycopg.connect(DB, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM lucidota_korpus.file_object")
            file_objects = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(DISTINCT file_uuid) AS n FROM lucidota_korpus.file_occurrence WHERE mtime IS NOT NULL")
            occurrence_mtime_files = cur.fetchone()["n"]
            cur.execute("SELECT processed_count FROM lucidota_korpus.chrono_replay_cursor WHERE cursor_name='chrono-ledger-daemon'")
            row = cur.fetchone()
            processed_count = row["processed_count"] if row else None
            cur.execute(
                """
                SELECT f.file_uuid::text, f.sha256, f.first_seen_path, f.first_seen_at::text, f.detail
                FROM lucidota_korpus.file_object f
                WHERE NOT EXISTS (
                    SELECT 1 FROM lucidota_korpus.file_occurrence o
                    WHERE o.file_uuid = f.file_uuid AND o.mtime IS NOT NULL
                )
                ORDER BY f.first_seen_at DESC
                LIMIT 50
                """
            )
            rows_without_mtime = list(cur.fetchall())
    delta = int(file_objects) - int(occurrence_mtime_files)
    explained = delta == len(rows_without_mtime) and delta == 1 and any((row.get("detail") or {}).get("smoke") == "phase_c_missed_replay" for row in rows_without_mtime)
    blocker = "" if explained else "MTIME_FREEZE_ROW_DELTA_UNEXPLAINED"
    report = {
        "schema": "lucidota.chrono.mtime_freeze_discrepancy.v1",
        "generated_at": iso_now(),
        "mtime_freeze_report": str(latest_mtime.relative_to(ROOT)),
        "chrono_processed_count": processed_count,
        "file_object_count": file_objects,
        "mtime_candidate_rows": mtime_report.get("candidate_file_rows"),
        "file_occurrence_files_with_mtime": occurrence_mtime_files,
        "delta_file_object_minus_mtime_candidates": int(file_objects) - int(mtime_report.get("candidate_file_rows", 0)),
        "rows_without_mtime_like_source": rows_without_mtime,
        "conclusion": "1-row delta is the Phase C missed-replay smoke file_object with no file_occurrence.mtime source; it is not eligible for mtime_snapshot_v1 freeze." if explained else "Unresolved mtime freeze row delta.",
        "execute_performed": False,
        "claims_inserted": 0,
        "dead_letter_mutated": False,
        "blocker": blocker,
    }
    out = OUT_DIR / f"chrono_mtime_freeze_discrepancy_{stamp()}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"MTIME_DISCREPANCY_REPORT={out.relative_to(ROOT)}")
    return 0 if not blocker else 4


if __name__ == "__main__":
    raise SystemExit(main())
