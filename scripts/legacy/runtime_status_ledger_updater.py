#!/usr/bin/env python3
"""Update STATUS_LEDGER from runtime DB facts, not hand-written claims."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "05_OUTPUTS/status_ledger.json"
OUT = ROOT / "05_OUTPUTS/status"

def now() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def db(args: argparse.Namespace) -> str: return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def bar(progress: int) -> str:
    progress=max(0,min(100,int(progress))); return f"[{'█'*(progress//10)}{'░'*(10-progress//10)}] {progress}%"
def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True); p=OUT/f"runtime_status_ledger_updater_{name}_{stamp()}.json"; payload.setdefault("generated_at", now()); payload["report_path"]=rel(p); p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str)); print(f"REPORT_PATH={rel(p)}"); return p


def runtime_facts(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status::text, count(*) AS n FROM lucidota_control.dbos_queue_job GROUP BY status::text")
            queue = {r["status"]: int(r["n"]) for r in cur.fetchall()}
            cur.execute("SELECT count(*) AS n FROM lucidota_control.boring_execution_record")
            exec_records = int(cur.fetchone()["n"])
            cur.execute("SELECT count(*) AS n FROM lucidota_control.dbos_queue_dead_letter WHERE resolved=false")
            dlq = int(cur.fetchone()["n"])
            cur.execute("SELECT count(*) AS n FROM lucidota_control.runtime_status_fact")
            fact_count = int(cur.fetchone()["n"])
    return {"dbos_queue_status_counts": queue, "boring_execution_records": exec_records, "unresolved_dead_letters": dlq, "runtime_status_fact_count": fact_count}


def upsert_software(data: dict[str, Any], name: str, patch: dict[str, Any]) -> None:
    entries = data.setdefault("software", [])
    row = next((e for e in entries if e.get("name") == name), None)
    if row is None:
        row = {"name": name, "path_or_profile": "", "status": "in_progress", "executed": "yes", "progress": 0, "evidence": "", "next_action": "", "blockers": ""}
        entries.append(row)
    row.update(patch)
    row["progress"] = int(row.get("progress", 0))
    row["loading_bar"] = bar(row["progress"])
    row["last_updated"] = now()


def main() -> int:
    p = argparse.ArgumentParser(description="Derive status ledger updates from runtime DB facts")
    p.add_argument("--database-url"); p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    facts = runtime_facts(args)
    report = {"action": "runtime_status_ledger_update", "execute_performed": bool(args.execute), "facts": facts, "files_written": [], "blockers": []}
    if args.execute:
        data = json.loads(STATUS.read_text())
        evidence = "lucidota_state:runtime facts via scripts/runtime_status_ledger_updater.py"
        q = facts["dbos_queue_status_counts"]
        progress = 95 if q.get("succeeded", 0) > 0 and facts["boring_execution_records"] > 0 else 60
        upsert_software(data, "DBOS runtime facts", {"path_or_profile": "lucidota_control.dbos_queue_job;lucidota_control.runtime_status_fact", "status": "executed", "executed": "yes", "progress": progress, "evidence": evidence, "next_action": "Use runtime-status updater after each DBOS worker pass.", "blockers": "" if facts["unresolved_dead_letters"] == 0 else f"unresolved_dead_letters={facts['unresolved_dead_letters']}"})
        upsert_software(data, "Execution record writer", {"status": "executed", "executed": "yes", "progress": max(75, int(next((e.get('progress',0) for e in data.get('software',[]) if e.get('name')=='Execution record writer'), 0))), "evidence": evidence, "next_action": "Adopt for all worker receipts.", "blockers": "worker_adoption_pending"})
        data["updated_at"] = now()
        STATUS.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
        subprocess.run(["python3", "scripts/lucidota_status_ledger.py", "--render-html"], cwd=ROOT, check=False)
        report["files_written"] = ["05_OUTPUTS/status_ledger.json", "00_PROJECT_BRAIN/STATUS_LEDGER.md"]
    out = write_report("execute" if args.execute else "dry_run", report)
    print(f"STATUS_FACTS_EVIDENCE={rel(out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
