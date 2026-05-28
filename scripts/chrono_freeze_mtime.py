#!/usr/bin/env python3
"""Dry-run/default mtime evidence freezer for Chrono-Ledger.

Default mode is dry-run. Execute mode appends temporal_claim rows only and never
mutates file_object/file_occurrence custody rows.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"psycopg required: {exc}") from exc

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "chrono_ledger"
EVIDENCE_SOURCE = "mtime_snapshot_v1"
EXTRACTOR = "chrono_freeze_mtime.py"
EXTRACTOR_VERSION = "mtime-freeze-v1"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def redact_url(url: str) -> str:
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        return scheme + "://<redacted>@" + rest.split("@", 1)[1]
    return url


def db_source(args_url: str | None) -> tuple[str, str]:
    if args_url:
        return args_url, "argument/redacted"
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"], "env/redacted"
    return "postgresql:///lucidota_storage", "default/redacted"


def table_exists(cur, regclass: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (regclass,))
    return cur.fetchone()["to_regclass"] is not None


def columns(cur, schema: str, table: str) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return list(cur.fetchall())


def main() -> int:
    parser = argparse.ArgumentParser(description="Chrono mtime freeze dry-run/append-only tool")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--execute", action="store_true", help="append mtime_snapshot_v1 temporal claims; default is dry-run")
    parser.add_argument("--limit", type=int, default=0, help="optional execute limit; 0 means no limit")
    args = parser.parse_args()

    database_url, source = db_source(args.database_url)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"chrono_mtime_freeze_dry_run_{stamp()}.json"
    blockers: list[str] = []
    inspected_tables: list[str] = []
    mtime_fields_found: list[dict[str, str]] = []
    candidate_file_rows = 0
    existing_claims = 0
    would_insert_claims = 0
    inserted_claims = 0
    file_object_found = False

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            file_object_found = table_exists(cur, "lucidota_korpus.file_object")
            temporal_claim_found = table_exists(cur, "lucidota_korpus.temporal_claim")
            file_occurrence_found = table_exists(cur, "lucidota_korpus.file_occurrence")
            for schema, table in [
                ("lucidota_korpus", "file_object"),
                ("lucidota_korpus", "file_occurrence"),
                ("lucidota_korpus", "temporal_claim"),
            ]:
                if table_exists(cur, f"{schema}.{table}"):
                    inspected_tables.append(f"{schema}.{table}")
                    for col in columns(cur, schema, table):
                        if "time" in col["column_name"].lower() or col["column_name"].lower() in {"mtime", "ctime", "atime"}:
                            mtime_fields_found.append({
                                "table": f"{schema}.{table}",
                                "column": col["column_name"],
                                "data_type": col["data_type"],
                            })
            if not file_object_found:
                blockers.append("lucidota_korpus.file_object_missing")
            if not temporal_claim_found:
                blockers.append("lucidota_korpus.temporal_claim_missing")
            if not file_occurrence_found:
                blockers.append("lucidota_korpus.file_occurrence_missing")

            if file_occurrence_found and temporal_claim_found:
                cur.execute(
                    """
                    WITH candidate AS (
                        SELECT DISTINCT ON (fo.file_uuid)
                               fo.file_uuid, fo.sha256, fo.first_seen_path, occ.mtime
                        FROM lucidota_korpus.file_object fo
                        JOIN lucidota_korpus.file_occurrence occ ON occ.file_uuid = fo.file_uuid
                        WHERE occ.mtime IS NOT NULL
                        ORDER BY fo.file_uuid, occ.mtime ASC
                    )
                    SELECT COUNT(*) AS candidate_file_rows,
                           COUNT(*) FILTER (
                               WHERE NOT EXISTS (
                                   SELECT 1 FROM lucidota_korpus.temporal_claim tc
                                   WHERE tc.file_uuid = candidate.file_uuid
                                     AND tc.evidence_source = %s
                               )
                           ) AS would_insert_claims
                    FROM candidate
                    """,
                    (EVIDENCE_SOURCE,),
                )
                row = cur.fetchone()
                candidate_file_rows = int(row["candidate_file_rows"] or 0)
                would_insert_claims = int(row["would_insert_claims"] or 0)
                cur.execute("SELECT COUNT(*) AS n FROM lucidota_korpus.temporal_claim WHERE evidence_source=%s", (EVIDENCE_SOURCE,))
                existing_claims = int(cur.fetchone()["n"] or 0)

            if args.execute:
                if blockers:
                    raise SystemExit("cannot execute with blockers: " + ",".join(blockers))
                sql = """
                    WITH candidate AS (
                        SELECT DISTINCT ON (fo.file_uuid)
                               fo.file_uuid, fo.sha256, fo.first_seen_path, occ.mtime
                        FROM lucidota_korpus.file_object fo
                        JOIN lucidota_korpus.file_occurrence occ ON occ.file_uuid = fo.file_uuid
                        WHERE occ.mtime IS NOT NULL
                          AND NOT EXISTS (
                              SELECT 1 FROM lucidota_korpus.temporal_claim tc
                              WHERE tc.file_uuid = fo.file_uuid
                                AND tc.evidence_source = %s
                          )
                        ORDER BY fo.file_uuid, occ.mtime ASC
                    ), limited AS (
                        SELECT * FROM candidate
                        LIMIT CASE WHEN %s > 0 THEN %s ELSE 2147483647 END
                    )
                    INSERT INTO lucidota_korpus.temporal_claim
                        (file_uuid, candidate_timestamp, evidence_source, trust_weight, raw_evidence,
                         extractor, extractor_version, source_path, source_sha256, detail)
                    SELECT file_uuid, mtime, %s, 0.10, mtime::text,
                           %s, %s, first_seen_path, sha256,
                           jsonb_build_object('freeze_tool', 'chrono_freeze_mtime.py', 'custody_mutated', false)
                    FROM limited
                    ON CONFLICT DO NOTHING
                """
                cur.execute(sql, (EVIDENCE_SOURCE, args.limit, args.limit, EVIDENCE_SOURCE, EXTRACTOR, EXTRACTOR_VERSION))
                inserted_claims = cur.rowcount
                conn.commit()
            else:
                conn.rollback()

    report = {
        "schema": "lucidota.chrono.mtime_freeze_report.v1",
        "generated_at": iso_now(),
        "database_url_source": source,
        "database_url_redacted": redact_url(database_url),
        "inspected_tables": inspected_tables,
        "file_object_table_found": "yes" if file_object_found else "no",
        "candidate_file_rows": candidate_file_rows,
        "mtime_fields_found": mtime_fields_found,
        "existing_mtime_snapshot_v1_claims": existing_claims,
        "would_insert_claims": would_insert_claims,
        "execute_performed": bool(args.execute),
        "inserted_claims": inserted_claims,
        "blockers": blockers,
        "law": "execute mode appends temporal_claim rows only; no file_object custody mutation",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({
        "report_path": str(report_path.relative_to(ROOT)),
        "execute_performed": bool(args.execute),
        "candidate_file_rows": candidate_file_rows,
        "existing_mtime_snapshot_v1_claims": existing_claims,
        "would_insert_claims": would_insert_claims,
        "blockers": blockers,
    }, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
