#!/usr/bin/env python3
"""DBOS workflow wrapper for Big Board snapshots.

Big Board stays read-only by default. This wrapper is the explicit workflow-feed
path: collect a local snapshot, persist a compact workflow_event, and return the
same compact report for UI/replay checks.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

import psycopg
from dbos import DBOS, DBOSConfig

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path("python3")
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")


@DBOS.step()
def collect_big_board_step() -> dict:
    proc = subprocess.run(
        [str(PY), str(ROOT / "scripts" / "lucidota_big_board.py"), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"big board exited {proc.returncode}")
    return json.loads(proc.stdout)


@DBOS.step()
def persist_event_step(run_id: str, snapshot: dict) -> dict:
    detail = {
        "overall": snapshot.get("bars", {}).get("overall", "unknown"),
        "lowest_phase_count": len(snapshot.get("bars", {}).get("phases", [])),
        "counters": snapshot.get("counters", {}),
        "gpu_status": snapshot.get("gpu", {}).get("status", "unknown"),
        "scraper_fleet": {
            "status": snapshot.get("scraper_fleet", {}).get("status", "unknown"),
            "local_scripts": len(snapshot.get("scraper_fleet", {}).get("local_scripts", [])),
            "authorized_adapters": len(snapshot.get("scraper_fleet", {}).get("authorized_adapters", [])),
        },
    }
    with psycopg.connect(STATE_DB) as conn:
        event_id = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event
              (workflow_id, run_id, phase, status, source, detail)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            RETURNING event_id
            """,
            (
                "big-board-event-feed",
                run_id,
                "big_board",
                "succeeded",
                "lucidota_dbos_big_board",
                json.dumps(detail, sort_keys=True),
            ),
        ).fetchone()[0]
        conn.commit()
    return {"event_id": str(event_id), "detail": detail}


@DBOS.workflow()
def big_board_feed_workflow(run_id: str) -> dict:
    snapshot = collect_big_board_step()
    event = persist_event_step(run_id, snapshot)
    return {
        "run_id": run_id,
        "event_id": event["event_id"],
        "overall": event["detail"]["overall"],
        "gpu_status": event["detail"]["gpu_status"],
        "scraper_status": event["detail"]["scraper_fleet"]["status"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-big-board")
    ap.add_argument("--run-id", default=f"big-board-{uuid.uuid4()}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    config: DBOSConfig = {
        "name": "lucidota-dbos-big-board",
        "system_database_url": STATE_DB,
    }
    DBOS(config=config)
    DBOS.launch()
    result = big_board_feed_workflow(args.run_id)
    report = {"ok": True, "dbos_big_board": result}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    DBOS.destroy(destroy_registry=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
