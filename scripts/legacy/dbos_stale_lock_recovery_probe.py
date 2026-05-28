#!/usr/bin/env python3
"""Create and recover a stale in-flight DBOS job."""
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
OUT = ROOT / "05_OUTPUTS/dbos"

def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def db(args: argparse.Namespace) -> str: return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True); p=OUT/f"dbos_stale_lock_recovery_probe_{stamp()}.json"; payload.setdefault("generated_at", now()); payload["report_path"]=rel(p); p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str)); print(f"REPORT_PATH={rel(p)}"); return p


def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return {"command": " ".join(cmd), "returncode": proc.returncode, "stdout": proc.stdout[-2500:], "stderr": proc.stderr[-2500:]}


def force_stale(args: argparse.Namespace, idem: str) -> str:
    payload = {"target_number": 11, "target_name": "In-flight recovery", "idempotency_key": idem, "handler": "noop", "message": "stale recovery probe"}
    run([sys.executable, "scripts/boring_beast.py", "enqueue", "--execute", "--priority", "-9998", "--payload-json", json.dumps(payload)])
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lucidota.actor_role='worker'")
            cur.execute(
                """
                UPDATE lucidota_control.dbos_queue_job
                SET status='running', locked_by='stale-lock-probe', locked_at=now() - interval '2 hours', leased_by='stale-lock-probe', lease_expires_at=now() - interval '90 minutes'
                WHERE queue_name='boring_beast' AND idempotency_key=%s
                RETURNING job_uuid::text
                """,
                (idem.lower(),),
            )
            job_uuid = cur.fetchone()["job_uuid"]
        conn.commit()
    return job_uuid


def job_state(args: argparse.Namespace, job_uuid: str) -> dict[str, Any]:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT job_uuid::text,status::text,locked_by,locked_at::text,error_kind,error_message FROM lucidota_control.dbos_queue_job WHERE job_uuid=%s::uuid", (job_uuid,))
            return dict(cur.fetchone())


def main() -> int:
    p = argparse.ArgumentParser(description="Prove stale running/locked jobs recover to queued")
    p.add_argument("--database-url"); p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        write_report({"action": "stale_lock_recovery_probe", "execute_performed": False, "blockers": ["execute_required"]}); return 2
    run([sys.executable, "scripts/boring_beast.py", "init-schema", "--execute"])
    idem = f"round4-stale-{stamp()}".lower()
    job_uuid = force_stale(args, idem)
    before = job_state(args, job_uuid)
    recover = run([sys.executable, "scripts/boring_beast.py", "recover-stale", "--execute", "--timeout-seconds", "60"])
    after = job_state(args, job_uuid)
    passed = before["status"] == "running" and after["status"] == "queued" and after["locked_by"] is None
    report = {"action": "stale_lock_recovery_probe", "execute_performed": True, "db_writes_performed": True, "graph_writes_performed": False, "job_uuid": job_uuid, "before": before, "recover_command": recover, "after": after, "pass": passed, "blockers": [] if passed else ["stale_job_not_recovered_to_queued"]}
    write_report(report)
    print("STALE_RECOVERY=" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
