#!/usr/bin/env python3
"""Replay/inspect workflow_event history for DBOS signoff and dispatch tests."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import psycopg

STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")


def replay(workflow: str | None, run_id: str | None, limit: int) -> dict[str, Any]:
    filters = []
    params: list[Any] = []
    if workflow:
        filters.append("workflow_id=%s")
        params.append(workflow)
    if run_id:
        filters.append("run_id=%s")
        params.append(run_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(limit)
    with psycopg.connect(STATE_DB) as conn:
        rows = conn.execute(
            f"""
            SELECT event_id, workflow_id, run_id, phase, status, source, detail, created_at
            FROM lucidota_control.workflow_event
            {where}
            ORDER BY created_at ASC
            LIMIT %s
            """,
            params,
        ).fetchall()
    keys = ["event_id", "workflow", "run_id", "phase", "status", "source", "detail", "created_at"]
    events = [dict(zip(keys, row)) for row in rows]
    terminal = next((e for e in reversed(events) if e["status"] in {"succeeded", "failed", "cancelled"}), None)
    return {
        "ok": True,
        "workflow": workflow,
        "run_id": run_id,
        "event_count": len(events),
        "terminal_status": terminal["status"] if terminal else "none",
        "events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-replay")
    ap.add_argument("--workflow")
    ap.add_argument("--run-id")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = replay(args.workflow, args.run_id, args.limit)
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(f"{report['event_count']} events; terminal={report['terminal_status']}")
        for e in report["events"]:
            print(f"{e['created_at']} {e['workflow']} {e['run_id']} {e['phase']} {e['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
