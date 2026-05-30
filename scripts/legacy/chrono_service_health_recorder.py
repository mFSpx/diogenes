#!/usr/bin/env python3
"""Record Chrono service health as runtime facts and update STATUS_LEDGER from facts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "chrono_ledger"
STATE_SCHEMA = ROOT / "06_SCHEMA/039_dbos_real_work_loop.sql"
SERVICE = "lucidota-chrono-ledger.service"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"chrono_service_health_recorder_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def storage_facts(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(storage_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH violations AS (
                  SELECT r.file_uuid
                  FROM lucidota_korpus.resolved_chrono_timeline_with_claim r
                  JOIN lucidota_korpus.temporal_claim tc ON tc.file_uuid = r.file_uuid
                  WHERE tc.trust_weight > r.confidence_score
                     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp < r.resolved_timestamp)
                )
                SELECT count(*) AS ranking_violations FROM violations
            """)
            ranking_violations = int(cur.fetchone()["ranking_violations"])
            cur.execute("""
                SELECT cursor_name, processed_count, last_replay_finished_at::text, updated_at::text
                FROM lucidota_korpus.chrono_replay_cursor
                WHERE cursor_name='chrono-ledger-daemon'
            """)
            cursor = dict(cur.fetchone() or {})
            cur.execute("SELECT count(*) AS n FROM lucidota_korpus.chrono_dead_letter WHERE resolved=false")
            dead_letters = int(cur.fetchone()["n"])
            cur.execute("""
                SELECT count(*) AS pending
                FROM lucidota_korpus.file_object f
                WHERE NOT EXISTS (SELECT 1 FROM lucidota_korpus.temporal_claim tc WHERE tc.file_uuid=f.file_uuid)
            """)
            pending = int(cur.fetchone()["pending"])
            cur.execute("SELECT count(*) AS claims, count(DISTINCT file_uuid) AS files FROM lucidota_korpus.temporal_claim")
            claims = dict(cur.fetchone())
    return {
        "ranking_violations": ranking_violations,
        "cursor": cursor,
        "dead_letter_unresolved": dead_letters,
        "pending_target_count": pending,
        "claims": claims,
    }


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(STATE_SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "schema": rel(STATE_SCHEMA),
    })
    return 0


def record(args: argparse.Namespace) -> int:
    is_active = run(["systemctl", "--user", "is-active", SERVICE])
    is_enabled = run(["systemctl", "--user", "is-enabled", SERVICE])
    status = run(["systemctl", "--user", "--no-pager", "status", SERVICE])
    facts = storage_facts(args)
    active = is_active["stdout"] == "active"
    enabled = is_enabled["stdout"] == "enabled"
    fact_payloads = {
        "service_active": {"active": active, "systemctl": is_active},
        "service_enabled": {"enabled": enabled, "systemctl": is_enabled},
        "ranking_violations": facts["ranking_violations"],
        "cursor_processed_count": int(facts["cursor"].get("processed_count") or 0),
        "dead_letter_unresolved": facts["dead_letter_unresolved"],
        "pending_target_count": facts["pending_target_count"],
        "temporal_claims_total": int(facts["claims"].get("claims") or 0),
        "files_covered": int(facts["claims"].get("files") or 0),
    }
    report = {
        "action": "record",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "service": SERVICE,
        "active": active,
        "enabled": enabled,
        "systemctl_status_summary": status["stdout"][-4000:],
        "facts": fact_payloads,
        "blockers": [],
    }
    report_path = write_report("execute" if args.execute else "dry_run", report)
    if args.execute:
        evidence_refs = [rel(report_path), "scripts/chrono_service_health_recorder.py", "systemctl --user is-active lucidota-chrono-ledger.service"]
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                for key, value in fact_payloads.items():
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.runtime_status_fact(subsystem, fact_key, fact_value, evidence_refs)
                        VALUES ('Chrono-Ledger', %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (subsystem, fact_key) DO UPDATE SET
                          fact_value=EXCLUDED.fact_value,
                          evidence_refs=EXCLUDED.evidence_refs,
                          derived_at=now()
                        """,
                        (key, json.dumps(value), json.dumps(evidence_refs)),
                    )
            conn.commit()
        subprocess.run([
            sys.executable, "scripts/lucidota_status_ledger.py",
            "--set", "Chrono-Ledger daemon/service",
            "--status", "verified" if active and enabled and facts["ranking_violations"] == 0 else "blocked",
            "--progress", "100" if active and enabled and facts["ranking_violations"] == 0 else "80",
            "--executed", "yes",
            "--evidence", rel(report_path),
            "--next", "Continue DBOS/CEP/Graph work while recording runtime facts each pass.",
            "--blocker", "" if active and enabled and facts["ranking_violations"] == 0 else "chrono_service_or_ranking_health_failed",
        ], cwd=ROOT, check=False)
        report["db_writes_performed"] = True
        report["status_ledger_updated"] = True
        report_path.write_text(json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"CHRONO_SERVICE_ACTIVE={str(active).lower()}")
    print(f"RANKING_VIOLATIONS={facts['ranking_violations']}")
    print(f"RUNTIME_FACTS_WRITTEN={str(bool(args.execute)).lower()}")
    return 0 if active and enabled and facts["ranking_violations"] == 0 else 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Chrono service health into runtime_status_fact and STATUS_LEDGER")
    parser.add_argument("--state-database-url")
    parser.add_argument("--storage-database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("record")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=record)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
