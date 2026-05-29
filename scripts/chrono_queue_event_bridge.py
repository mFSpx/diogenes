#!/usr/bin/env python3
"""Append ABSURD queue events into Chrono as temporal evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = [
    ROOT / "06_SCHEMA/058_chrono_queue_event_bridge.sql",
    ROOT / "06_SCHEMA/067_chrono_queue_event_bridge_automation.sql",
]
OUT = ROOT / "05_OUTPUTS/chrono_ledger"

def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def sha_obj(obj: Any) -> str: return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
def state_db(a: argparse.Namespace) -> str: return a.state_database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def storage_db(a: argparse.Namespace) -> str: return a.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True); p=OUT/f"chrono_queue_event_bridge_{name}_{stamp()}.json"; payload.setdefault("generated_at", now()); payload["report_path"]=rel(p); p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str)); print(f"REPORT_PATH={rel(p)}"); return p


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text())
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "execute_performed": bool(args.execute),
        "schemas": [rel(s) for s in SCHEMAS],
    })
    return 0


def fetch_events(args: argparse.Namespace) -> list[dict[str, Any]]:
    with psycopg.connect(state_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.queue_event_uuid::text, e.job_uuid::text, e.queue_name, e.event_kind, e.event_source, e.detail, e.created_at::text
                FROM lucidota_control.absurd_queue_event e
                WHERE NOT EXISTS (
                  SELECT 1 FROM lucidota_control.chrono_queue_event_bridge_receipt r
                  WHERE r.queue_event_uuid=e.queue_event_uuid
                )
                ORDER BY e.created_at DESC LIMIT %s
                """,
                (args.limit,),
            )
            return [dict(r) for r in cur.fetchall()]


def append_event_batch(args: argparse.Namespace, events: list[dict[str, Any]]) -> dict[str, Any]:
    inserted = 0
    reused = 0
    receipt_conflicts = 0
    receipts: list[dict[str, Any]] = []
    if args.execute and events:
        with psycopg.connect(storage_db(args)) as storage, psycopg.connect(state_db(args)) as state:
            with storage.cursor() as scur, state.cursor() as qcur:
                for ev in events:
                    payload_sha = sha_obj(ev)
                    scur.execute(
                        """
                        SELECT claim_uuid::text
                        FROM lucidota_korpus.temporal_claim
                        WHERE evidence_source='absurd_queue_event_bridge'
                          AND source_sha256=%s
                        ORDER BY created_at ASC
                        LIMIT 1
                        """,
                        (payload_sha,),
                    )
                    existing = scur.fetchone()
                    if existing:
                        claim_uuid = existing[0]
                        reused += 1
                    else:
                        scur.execute(
                            """
                            INSERT INTO lucidota_korpus.temporal_claim(
                              candidate_timestamp, evidence_source, trust_weight, raw_evidence,
                              extractor, extractor_version, source_path, source_sha256, detail
                            ) VALUES (now(), 'absurd_queue_event_bridge', 0.55, %s, 'chrono_queue_event_bridge', 'v1',
                                      'lucidota_control.absurd_queue_event', %s, %s::jsonb)
                            RETURNING claim_uuid::text
                            """,
                            (json.dumps({"queue_event_uuid": ev["queue_event_uuid"], "event_kind": ev["event_kind"]}), payload_sha, json.dumps(ev)),
                        )
                        claim_uuid = scur.fetchone()[0]
                        inserted += 1
                    qcur.execute(
                        """
                        INSERT INTO lucidota_control.chrono_queue_event_bridge_receipt(queue_event_uuid, chrono_claim_uuid, event_payload_sha256, detail)
                        VALUES (%s::uuid,%s::uuid,%s,%s::jsonb)
                        ON CONFLICT (queue_event_uuid, evidence_source) DO NOTHING
                        RETURNING bridge_uuid::text
                        """,
                        (ev["queue_event_uuid"], claim_uuid, payload_sha, json.dumps({"script": "scripts/chrono_queue_event_bridge.py"})),
                    )
                    row = qcur.fetchone()
                    if row:
                        receipts.append({"queue_event_uuid": ev["queue_event_uuid"], "chrono_claim_uuid": claim_uuid, "bridge_uuid": row[0]})
                    else:
                        receipt_conflicts += 1
            storage.commit(); state.commit()
    return {
        "events_seen": len(events),
        "claims_inserted": inserted,
        "existing_claims_reused": reused,
        "receipts_inserted": len(receipts),
        "receipt_conflicts": receipt_conflicts,
        "receipts": receipts,
    }


def append_events(args: argparse.Namespace) -> int:
    events = fetch_events(args)
    batch = append_event_batch(args, events)
    report = {
        "action": "append_events",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        **batch,
        "blockers": [] if events else ["no_unbridged_queue_events"],
    }
    write_report("append_execute" if args.execute else "append_dry_run", report)
    print(f"CHRONO_QUEUE_EVENTS_SEEN={len(events)}")
    print(f"CHRONO_CLAIMS_INSERTED={batch['claims_inserted'] if args.execute else 0}")
    print(f"CHRONO_EXISTING_CLAIMS_REUSED={batch['existing_claims_reused'] if args.execute else 0}")
    print(f"CHRONO_RECEIPTS_INSERTED={batch['receipts_inserted'] if args.execute else 0}")
    return 0 if events or not args.require_events else 2


def bridge_loop(args: argparse.Namespace) -> int:
    started = now()
    run_uuid = None
    totals = {
        "iterations_completed": 0,
        "events_seen": 0,
        "claims_inserted": 0,
        "existing_claims_reused": 0,
        "receipts_inserted": 0,
        "receipt_conflicts": 0,
        "idle_iterations": 0,
    }
    iteration_reports: list[dict[str, Any]] = []
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.chrono_queue_event_bridge_run(run_mode, iterations_requested, detail)
                    VALUES ('execute', %s, %s::jsonb)
                    RETURNING run_uuid::text
                    """,
                    (args.iterations, json.dumps({"script": "scripts/chrono_queue_event_bridge.py", "command": "loop"})),
                )
                run_uuid = cur.fetchone()[0]
            conn.commit()
    for idx in range(args.iterations):
        events = fetch_events(args)
        batch = append_event_batch(args, events)
        totals["iterations_completed"] += 1
        for key in ("events_seen", "claims_inserted", "existing_claims_reused", "receipts_inserted", "receipt_conflicts"):
            totals[key] += int(batch[key])
        if not events:
            totals["idle_iterations"] += 1
        else:
            totals["idle_iterations"] = 0
        iteration_reports.append({"iteration": idx + 1, **{k: v for k, v in batch.items() if k != "receipts"}, "receipt_sample": batch["receipts"][:3]})
        if args.stop_when_idle and totals["idle_iterations"] >= args.idle_stop_after:
            break
        if idx + 1 < args.iterations and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    report = {
        "action": "loop",
        "started_at": started,
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "run_uuid": run_uuid,
        "iterations_requested": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "stop_when_idle": args.stop_when_idle,
        **totals,
        "iterations": iteration_reports,
        "blockers": [] if totals["events_seen"] else ["no_unbridged_queue_events"],
    }
    report_path = write_report("loop_execute" if args.execute else "loop_dry_run", report)
    if args.execute and run_uuid:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE lucidota_control.chrono_queue_event_bridge_run
                    SET iterations_completed=%s, events_seen=%s, claims_inserted=%s,
                        existing_claims_reused=%s, receipts_inserted=%s, idle_iterations=%s,
                        finished_at=now(), report_path=%s, detail=detail || %s::jsonb
                    WHERE run_uuid=%s::uuid
                    """,
                    (
                        totals["iterations_completed"], totals["events_seen"], totals["claims_inserted"],
                        totals["existing_claims_reused"], totals["receipts_inserted"], totals["idle_iterations"],
                        rel(report_path), json.dumps({"receipt_conflicts": totals["receipt_conflicts"]}), run_uuid,
                    ),
                )
            conn.commit()
    print(f"CHRONO_BRIDGE_RUN_UUID={run_uuid or ''}")
    print(f"CHRONO_QUEUE_EVENTS_SEEN={totals['events_seen']}")
    print(f"CHRONO_CLAIMS_INSERTED={totals['claims_inserted'] if args.execute else 0}")
    print(f"CHRONO_RECEIPTS_INSERTED={totals['receipts_inserted'] if args.execute else 0}")
    return 0


def main() -> int:
    p=argparse.ArgumentParser(description="Bridge ABSURD queue events into Chrono temporal claims")
    p.add_argument("--state-database-url"); p.add_argument("--storage-database-url")
    sub=p.add_subparsers(dest="cmd", required=True)
    sp=sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp=sub.add_parser("append"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--limit", type=int, default=20); sp.add_argument("--require-events", action="store_true"); sp.set_defaults(func=append_events)
    sp=sub.add_parser("loop")
    sp.add_argument("--execute", action="store_true")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--iterations", type=int, default=3)
    sp.add_argument("--sleep-seconds", type=float, default=1.0)
    sp.add_argument("--stop-when-idle", action="store_true", default=True)
    sp.add_argument("--idle-stop-after", type=int, default=1)
    sp.set_defaults(func=bridge_loop)
    a=p.parse_args(); return a.func(a)
if __name__ == "__main__": raise SystemExit(main())
