#!/usr/bin/env python3
"""Register ABSURD worker contracts for the krampus_ingest pipeline.

Upserts three rows into lucidota_control.absurd_worker_contract:
  - krampus_ingest / intake       / krampus_intake_v1
  - krampus_ingest / route_extract / krampus_extract_v1
  - krampus_ingest / stage        / krampus_stage_v1

Conflict target: worker_key (the only unique constraint besides PK).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]

CONTRACTS = [
    {
        "worker_key": "krampus_intake_v1",
        "queue_name": "krampus_ingest",
        "script_path": "scripts/krampus_intake_worker.py",
        "job_kind": "intake",
        "status": "implemented",
    },
    {
        "worker_key": "krampus_extract_v1",
        "queue_name": "krampus_ingest",
        "script_path": "scripts/krampus_route_extract_worker.py",
        "job_kind": "route_extract",
        "status": "implemented",
    },
    {
        "worker_key": "krampus_stage_v1",
        "queue_name": "krampus_ingest",
        "script_path": "scripts/krampus_stage_worker.py",
        "job_kind": "stage",
        "status": "implemented",
    },
]


def db_url() -> str:
    return (
        os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def script_exists(script_path: str) -> bool:
    p = Path(script_path)
    if not p.is_absolute():
        p = ROOT / p
    return p.exists() and p.is_file()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dry_run() -> int:
    print(f"[dry-run] database: {db_url()}")
    print(f"[dry-run] would UPSERT {len(CONTRACTS)} contract row(s):\n")
    blockers: list[str] = []
    for c in CONTRACTS:
        exists = script_exists(c["script_path"])
        marker = "OK" if exists else "MISSING_SCRIPT"
        if not exists:
            blockers.append(f"script_missing:{c['script_path']}")
        print(
            f"  worker_key={c['worker_key']!r:30s}  queue={c['queue_name']!r:20s}"
            f"  job_kind={c['job_kind']!r:16s}  script={marker}"
        )
    if blockers:
        print(f"\n[dry-run] BLOCKERS: {blockers}")
        return 1
    print("\n[dry-run] no blockers. pass --execute to write.")
    return 0


def execute() -> int:
    blockers: list[str] = []
    for c in CONTRACTS:
        if not script_exists(c["script_path"]):
            blockers.append(f"script_missing:{c['script_path']}")
    if blockers:
        print(f"BLOCKED: {blockers}", file=sys.stderr)
        return 1

    url = db_url()
    upserted: list[str] = []
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            for c in CONTRACTS:
                input_contract = json.dumps({"job_kind": c["job_kind"]})
                output_contract = json.dumps({"status": "required"})
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_worker_contract
                      (worker_key, queue_name, script_path,
                       input_contract, output_contract,
                       idempotency_rule, retry_policy, dead_letter_policy,
                       canonical_graph_write_allowed, status,
                       evidence_refs, detail)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb,
                            'deduplicate_by_idempotency_key',
                            'retry_up_to_max_attempts',
                            'dead_letter_on_final_attempt',
                            false, %s,
                            '[]'::jsonb, %s::jsonb)
                    ON CONFLICT (worker_key) DO UPDATE
                      SET queue_name   = EXCLUDED.queue_name,
                          script_path  = EXCLUDED.script_path,
                          input_contract = EXCLUDED.input_contract,
                          status       = EXCLUDED.status,
                          updated_at   = now()
                    RETURNING worker_key, queue_name, status
                    """,
                    (
                        c["worker_key"],
                        c["queue_name"],
                        c["script_path"],
                        input_contract,
                        output_contract,
                        c["status"],
                        json.dumps({"registered_by": "krampus_register_contracts.py", "registered_at": now_iso()}),
                    ),
                )
                row = cur.fetchone()
                upserted.append(str(row))
                print(f"UPSERTED: {row}")
        conn.commit()

    print(f"\nRegistered {len(upserted)} contract(s). Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Register krampus_ingest worker contracts.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.execute:
        return execute()
    return dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
