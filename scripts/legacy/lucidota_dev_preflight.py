#!/usr/bin/env python3
"""Runtime preflight for LUCIDOTA development discipline.

Checks the minimum live gates before coding/worker passes:
- TICKLETRUNK manifest exists and parses
- status ledger checker passes
- direct canonical graph writes are blocked
- Boring Beast E2E command exists

With --execute it records a machine status fact in lucidota_state only.
It does not mutate canonical graph or storage evidence tables.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/preflight"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def control_db(args: argparse.Namespace) -> str:
    return args.control_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def run_cmd(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "passed": proc.returncode == 0,
    }


def check_tickletrunk() -> dict[str, Any]:
    path = ROOT / "00_PROJECT_BRAIN/TICKLETRUNK.json"
    result: dict[str, Any] = {"path": rel(path), "exists": path.exists(), "parsed": False, "total_tools": None}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        result.update({"parsed": True, "total_tools": data.get("total_tools"), "read_first_rule": bool(data.get("read_first_rule"))})
    return result


def check_graph_barrier(args: argparse.Namespace) -> dict[str, Any]:
    result = {"direct_write_blocked": False, "error": "", "canonical_graph_writes_performed": False}
    try:
        with psycopg.connect(storage_db(args)) as conn:
            with conn.cursor() as cur:
                before = None
                cur.execute("SELECT count(*) FROM lucidota_go.graph_item")
                before = int(cur.fetchone()[0])
                try:
                    cur.execute(
                        """
                        INSERT INTO lucidota_go.graph_item(term,label,status,location_at_on_graph,payload)
                        VALUES ('CLAIM','direct graph write probe','staged','lucidota_dev_preflight.py','{"probe":true}'::jsonb)
                        """
                    )
                    conn.rollback()
                    result["direct_write_blocked"] = False
                    result["error"] = "direct insert unexpectedly succeeded"
                except Exception as exc:
                    conn.rollback()
                    result["direct_write_blocked"] = True
                    result["error"] = str(exc).splitlines()[0][:500]
                with conn.cursor() as cur2:
                    cur2.execute("SELECT count(*) FROM lucidota_go.graph_item")
                    after = int(cur2.fetchone()[0])
                result["canonical_graph_writes_performed"] = before != after
    except Exception as exc:
        result["error"] = f"graph_barrier_check_failed:{type(exc).__name__}:{exc}"
    return result


def write_status_fact(args: argparse.Namespace, report: dict[str, Any]) -> None:
    with psycopg.connect(control_db(args)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.runtime_status_fact(subsystem, fact_key, fact_value, evidence_refs)
                VALUES ('lucidota_dev_preflight','latest',%s::jsonb,%s::jsonb)
                ON CONFLICT (subsystem, fact_key)
                DO UPDATE SET fact_value=EXCLUDED.fact_value, evidence_refs=EXCLUDED.evidence_refs, derived_at=now()
                """,
                (json.dumps(report), json.dumps([report["report_path"], "scripts/lucidota_dev_preflight.py"])),
            )
        conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LUCIDOTA development/runtime discipline preflight")
    parser.add_argument("--execute", action="store_true", help="record runtime_status_fact in control DB")
    parser.add_argument("--control-database-url")
    parser.add_argument("--storage-database-url")
    args = parser.parse_args()

    status_check = run_cmd(["python3", "scripts/lucidota_status_ledger.py", "--check"])
    report = {
        "schema": "lucidota.dev_preflight.v1",
        "generated_at": now_iso(),
        "execute_performed": bool(args.execute),
        "tickletrunk": check_tickletrunk(),
        "status_ledger_check": status_check,
        "boring_beast_exists": (ROOT / "scripts/boring_beast.py").exists(),
        "graph_write_barrier": check_graph_barrier(args),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "blockers": [],
    }
    if not report["tickletrunk"].get("exists") or not report["tickletrunk"].get("parsed"):
        report["blockers"].append("tickletrunk_manifest_missing_or_invalid")
    if not status_check["passed"]:
        report["blockers"].append("status_ledger_check_failed")
    if not report["boring_beast_exists"]:
        report["blockers"].append("boring_beast_missing")
    if not report["graph_write_barrier"].get("direct_write_blocked"):
        report["blockers"].append("direct_graph_write_not_blocked")
    if report["graph_write_barrier"].get("canonical_graph_writes_performed"):
        report["blockers"].append("graph_barrier_probe_mutated_graph")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"lucidota_dev_preflight_{'execute' if args.execute else 'dry_run'}_{stamp()}.json"
    report["report_path"] = rel(out)
    if args.execute and not report["blockers"]:
        write_status_fact(args, report)
        report["db_writes_performed"] = True
    out.write_text(json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={rel(out)}")
    print("PREFLIGHT_STATUS=" + ("PASS" if not report["blockers"] else "BLOCKED"))
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
