#!/usr/bin/env python3
"""DBOS scheduled watcher v0.

Seeds a tiny local schedule table and records due workflow ticks. It does not
daemonize; cron/systemd can call this. The test path proves due detection,
locking-by-next-run update, and workflow_event emission.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import psycopg
from dbos import DBOS, DBOSConfig

STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")

SCHEDULE_SQL = """
CREATE TABLE IF NOT EXISTS lucidota_control.workflow_schedule (
  schedule_id text PRIMARY KEY,
  workflow_name text NOT NULL,
  cadence_seconds integer NOT NULL CHECK (cadence_seconds > 0),
  enabled boolean NOT NULL DEFAULT true,
  next_run_at timestamptz NOT NULL DEFAULT now(),
  last_run_at timestamptz,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
);
"""

DEFAULTS = [
    ("big-board-heartbeat", "big-board-event-feed", 900),
    ("model-governor-heartbeat", "model-governor", 1800),
    ("drive-map-daily-check", "drive-map-workflow", 86400),
]


@DBOS.step()
def seed_step() -> dict[str, Any]:
    with psycopg.connect(STATE_DB) as conn:
        conn.execute(SCHEDULE_SQL)
        for sid, workflow, cadence in DEFAULTS:
            conn.execute(
                """
                INSERT INTO lucidota_control.workflow_schedule
                  (schedule_id, workflow_name, cadence_seconds, next_run_at, detail)
                VALUES (%s, %s, %s, now(), %s::jsonb)
                ON CONFLICT (schedule_id) DO UPDATE SET
                  workflow_name=EXCLUDED.workflow_name,
                  cadence_seconds=EXCLUDED.cadence_seconds,
                  enabled=true,
                  detail=EXCLUDED.detail,
                  updated_at=now()
                """,
                (sid, workflow, cadence, json.dumps({"seeded_by": "lucidota_dbos_watcher"})),
            )
        conn.commit()
    return {"seeded": len(DEFAULTS)}


@DBOS.step()
def tick_step(limit: int) -> dict[str, Any]:
    with psycopg.connect(STATE_DB) as conn:
        conn.execute(SCHEDULE_SQL)
        rows = conn.execute(
            """
            SELECT schedule_id, workflow_name, cadence_seconds
            FROM lucidota_control.workflow_schedule
            WHERE enabled=true AND next_run_at <= now()
            ORDER BY next_run_at ASC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        ticks = []
        for schedule_id, workflow, cadence in rows:
            run_id = f"scheduled-{schedule_id}"
            event_id = conn.execute(
                """
                INSERT INTO lucidota_control.workflow_event
                  (workflow_id, run_id, phase, status, source, detail)
                VALUES (%s, %s, 'scheduled_watch', 'queued', 'lucidota_dbos_watcher', %s::jsonb)
                RETURNING event_id
                """,
                (workflow, run_id, json.dumps({"schedule_id": schedule_id, "cadence_seconds": cadence})),
            ).fetchone()[0]
            conn.execute(
                """
                UPDATE lucidota_control.workflow_schedule
                SET last_run_at=now(),
                    next_run_at=now() + make_interval(secs => cadence_seconds),
                    updated_at=now()
                WHERE schedule_id=%s
                """,
                (schedule_id,),
            )
            ticks.append({"schedule_id": schedule_id, "workflow": workflow, "event_id": str(event_id)})
        conn.commit()
    return {"due": len(ticks), "ticks": ticks}


@DBOS.workflow()
def watcher_workflow(seed: bool, limit: int) -> dict[str, Any]:
    seeded = seed_step() if seed else {"seeded": 0}
    ticked = tick_step(limit)
    return {"seed": seeded, "tick": ticked}


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-watcher")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    DBOS(config={"name": "lucidota-dbos-watcher", "system_database_url": STATE_DB})
    DBOS.launch()
    result = watcher_workflow(args.seed, args.limit)
    DBOS.destroy(destroy_registry=True)
    report = {"ok": True, "watcher": result}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
