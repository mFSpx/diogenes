#!/usr/bin/env python3
"""Write Chrono-Ledger service activation report. Read-only except report file."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception as exc:
    raise SystemExit(f"psycopg required: {exc}")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "chrono_ledger"
SERVICE = "lucidota-chrono-ledger.service"
DB = os.environ.get("DATABASE_URL", "postgresql:///lucidota_storage")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {"cmd": cmd, "rc": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def query_report() -> dict:
    with psycopg.connect(DB, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH violations AS (
                  SELECT r.file_uuid
                  FROM lucidota_korpus.resolved_chrono_timeline r
                  JOIN lucidota_korpus.temporal_claim tc ON tc.file_uuid = r.file_uuid
                  WHERE tc.trust_weight > r.confidence_score
                     OR (tc.trust_weight = r.confidence_score AND tc.candidate_timestamp < r.resolved_timestamp)
                ) SELECT COUNT(*) AS ranking_violations FROM violations
            """)
            ranking_violations = cur.fetchone()["ranking_violations"]
            cur.execute("""
                SELECT cursor_name,last_file_uuid::text,last_file_first_seen_at::text,processed_count,
                       last_replay_started_at::text,last_replay_finished_at::text,updated_at::text
                FROM lucidota_korpus.chrono_replay_cursor
                ORDER BY cursor_name
            """)
            cursor_rows = list(cur.fetchall())
            cur.execute("SELECT resolved, COUNT(*) AS count FROM lucidota_korpus.chrono_dead_letter GROUP BY resolved ORDER BY resolved")
            dead_letters = list(cur.fetchall())
            cur.execute("""
                SELECT COUNT(*) AS pending_target_count
                FROM lucidota_korpus.file_object f
                WHERE NOT EXISTS (SELECT 1 FROM lucidota_korpus.temporal_claim tc WHERE tc.file_uuid=f.file_uuid)
            """)
            pending_target_count = cur.fetchone()["pending_target_count"]
            cur.execute("SELECT COUNT(*) AS total_claims, COUNT(DISTINCT file_uuid) AS files_covered FROM lucidota_korpus.temporal_claim")
            claims = cur.fetchone()
            return {
                "ranking_violations": ranking_violations,
                "replay_cursor_status": cursor_rows,
                "dead_letter_status": dead_letters,
                "pending_target_count": pending_target_count,
                "total_claims": claims["total_claims"],
                "files_covered": claims["files_covered"],
            }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status = run(["systemctl", "--user", "--no-pager", "status", SERVICE])
    is_active = run(["systemctl", "--user", "is-active", SERVICE])
    is_enabled = run(["systemctl", "--user", "is-enabled", SERVICE])
    db = query_report()
    latest_phase_c = sorted((ROOT / "05_OUTPUTS" / "chrono_ledger").glob("chrono_ledger_phase_c_report_*.json"))[-1]
    active = is_active["stdout"].strip() == "active"
    enabled = is_enabled["stdout"].strip() == "enabled"
    blocker = "" if active else f"service is not active: is-active={is_active['stdout'] or is_active['stderr']}"
    report = {
        "schema": "lucidota.chrono_ledger.service_activation_report.v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "service_unit_name": SERVICE,
        "enabled": enabled,
        "active": active,
        "systemctl_status_summary": status["stdout"],
        "systemctl_status_rc": status["rc"],
        "ranking_violations_after_service_start": db["ranking_violations"],
        "replay_cursor_status": db["replay_cursor_status"],
        "dead_letter_table_status": db["dead_letter_status"],
        "pending_target_count": db["pending_target_count"],
        "total_claims": db["total_claims"],
        "files_covered": db["files_covered"],
        "latest_phase_c_report_path": str(latest_phase_c.relative_to(ROOT)),
        "blocker": blocker,
    }
    path = OUT / f"chrono_ledger_service_activation_report_{utc_stamp()}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({
        "path": str(path.relative_to(ROOT)),
        "active": active,
        "enabled": enabled,
        "ranking_violations": db["ranking_violations"],
        "pending_target_count": db["pending_target_count"],
        "blocker": blocker,
    }, indent=2, sort_keys=True))
    return 0 if active and enabled and db["ranking_violations"] == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
