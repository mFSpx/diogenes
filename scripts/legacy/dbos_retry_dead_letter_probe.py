#!/usr/bin/env python3
"""Probe DBOS retry/dead-letter behavior with a bounded failing job."""
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


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
    return {"command": " ".join(cmd), "returncode": proc.returncode, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:]}


def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"dbos_retry_dead_letter_probe_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def latest_job(idem: str, args: argparse.Namespace) -> dict[str, Any] | None:
    idem_norm = "-".join(str(idem).strip().lower().split())
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.job_uuid::text, j.status::text, j.attempt_count, j.max_attempts,
                       j.error_kind, j.error_message,
                       EXISTS(SELECT 1 FROM lucidota_control.dbos_queue_dead_letter dlq WHERE dlq.job_uuid=j.job_uuid AND dlq.resolved=false) AS unresolved_dlq
                FROM lucidota_control.dbos_queue_job j
                WHERE j.queue_name='boring_beast' AND j.idempotency_key=%s
                ORDER BY j.created_at DESC LIMIT 1
                """,
                (idem_norm,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def probe(args: argparse.Namespace) -> int:
    idem = f"round4-retry-dlq-{stamp()}"
    payload = {
        "target_number": 6,
        "target_name": "Retry and dead-letter behavior",
        "idempotency_key": idem,
        "handler": "fail_once",
        "fail": True,
        "files_changed": ["scripts/dbos_retry_dead_letter_probe.py"],
        "validation_commands": ["python3 scripts/dbos_retry_dead_letter_probe.py --execute"],
    }
    cmds = [
        run([sys.executable, "scripts/boring_beast.py", "init-schema", "--execute"]),
        run([sys.executable, "scripts/boring_beast.py", "enqueue", "--execute", "--priority", "-9999", "--max-attempts", "1", "--payload-json", json.dumps(payload)]),
        run([sys.executable, "scripts/boring_beast.py", "worker-once", "--execute", "--worker-id", "retry-dlq-probe"]),
        run([sys.executable, "scripts/boring_beast.py", "retry-failed", "--execute"]),
    ]
    job = latest_job(idem, args)
    passed = bool(job and job["status"] == "dead_lettered" and int(job["attempt_count"]) == 1 and job["unresolved_dlq"])
    report = {
        "action": "probe_retry_dead_letter",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "idempotency_key": idem,
        "commands": cmds,
        "job": job,
        "pass": passed,
        "blockers": [] if passed else ["bad_job_did_not_dead_letter_or_loop_bound_failed"],
    }
    write_report(report)
    print("RETRY_DLQ_PROBE=" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Create a bounded failing job and prove DLQ behavior")
    p.add_argument("--state-database-url")
    p.add_argument("--execute", action="store_true", help="required acknowledgement; probe writes a test job/DLQ row")
    args = p.parse_args()
    if not args.execute:
        write_report({"action": "probe_retry_dead_letter", "execute_performed": False, "blockers": ["execute_required_for_probe"]})
        return 2
    return probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
