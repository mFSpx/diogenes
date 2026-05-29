#!/usr/bin/env python3
"""DBOS external-write draft gate.

Creates a pending governance gate for any proposed external write without
performing the write. Approval can be handled by lucidota_dbos_signoff.py.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import Any

import psycopg
from dbos import DBOS, DBOSConfig

STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")


def emit(run_id: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DB) as conn:
        event_id = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event
              (workflow_id, run_id, phase, status, source, detail)
            VALUES ('external-write-draft-gate', %s, 'external_write_gate', %s,
                    'lucidota_dbos_external_draft', %s::jsonb)
            RETURNING event_id
            """,
            (run_id, status, json.dumps(detail, sort_keys=True)),
        ).fetchone()[0]
        conn.commit()
    return str(event_id)


@DBOS.step()
def create_draft_gate_step(run_id: str, target: str, draft: str, risk: str, requested_by: str) -> dict[str, Any]:
    with psycopg.connect(STATE_DB) as conn:
        gate_id = conn.execute(
            """
            INSERT INTO lucidota_control.governance_gate
              (workflow_id, run_id, action_kind, requested_by, target, risk_level,
               policy_mode, approval_status, rationale, evidence)
            VALUES ('external-write-draft-gate', %s, 'external_write', %s, %s, %s,
                    'user_controlled', 'pending',
                    'external write draft requires explicit approval before send/apply',
                    %s::jsonb)
            RETURNING gate_id
            """,
            (
                run_id,
                requested_by,
                target,
                risk,
                json.dumps({"draft": draft, "write_performed": False}, sort_keys=True),
            ),
        ).fetchone()[0]
        conn.commit()
    event_id = emit(run_id, "waiting_user", {"gate_id": str(gate_id), "target": target, "risk": risk})
    return {"gate_id": str(gate_id), "event_id": event_id, "target": target, "write_performed": False}


@DBOS.workflow()
def external_draft_workflow(run_id: str, target: str, draft: str, risk: str, requested_by: str) -> dict[str, Any]:
    return create_draft_gate_step(run_id, target, draft, risk, requested_by)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-external-draft")
    ap.add_argument("--run-id", default=f"external-draft-{uuid.uuid4()}")
    ap.add_argument("--target", default="external://test")
    ap.add_argument("--draft", default="TEST DRAFT ONLY - DO NOT SEND")
    ap.add_argument("--risk", choices=["low", "medium", "high", "critical"], default="medium")
    ap.add_argument("--requested-by", default="operator")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    DBOS(config={"name": "lucidota-dbos-external-draft", "system_database_url": STATE_DB})
    DBOS.launch()
    result = external_draft_workflow(args.run_id, args.target, args.draft, args.risk, args.requested_by)
    DBOS.destroy(destroy_registry=True)
    report = {"ok": True, "external_draft": result}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
