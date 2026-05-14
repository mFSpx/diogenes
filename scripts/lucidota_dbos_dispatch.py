#!/usr/bin/env python3
"""DBOS workflow dispatcher with signoff + retry.

This is the one-hour test lane:
1. pick a registered workflow,
2. require or auto-create operator signoff,
3. run the registered command as a DBOS workflow step,
4. emit workflow_event rows for queued/running/retry/succeeded/failed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
from dbos import DBOS, DBOSConfig

ROOT = Path(__file__).resolve().parents[1]
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
REGISTRY_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)


def jdump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def emit_event(workflow: str, run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DB) as conn:
        event_id = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event
              (workflow_id, run_id, phase, status, source, detail)
            VALUES (%s, %s, %s, %s, 'lucidota_dbos_dispatch', %s::jsonb)
            RETURNING event_id
            """,
            (workflow, run_id, phase, status, jdump(detail)),
        ).fetchone()[0]
        conn.commit()
    return str(event_id)


def registry_row(workflow: str) -> dict[str, Any]:
    with psycopg.connect(STATE_DB) as conn:
        conn.execute(REGISTRY_SCHEMA.read_text())
        row = conn.execute(
            """
            SELECT workflow_name, owner, phase, status, command, inputs, outputs, notes
            FROM lucidota_control.workflow_registry
            WHERE workflow_name=%s
            """,
            (workflow,),
        ).fetchone()
        conn.commit()
    if not row:
        raise ValueError(f"unknown registered workflow: {workflow}")
    keys = ["workflow", "owner", "phase", "registry_status", "command", "inputs", "outputs", "notes"]
    data = dict(zip(keys, row))
    if data["registry_status"] not in {"active", "prototype"}:
        raise ValueError(f"workflow {workflow} is {data['registry_status']}, not dispatchable")
    return data


def ensure_signoff(workflow: str, run_id: str, auto_approve: bool, signer: str) -> dict[str, Any]:
    with psycopg.connect(STATE_DB) as conn:
        approved = conn.execute(
            """
            SELECT gate_id, approval_status
            FROM lucidota_control.governance_gate
            WHERE workflow_id=%s AND run_id=%s AND approval_status='approved'
            ORDER BY decided_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """,
            (workflow, run_id),
        ).fetchone()
        if approved:
            return {"status": "approved", "gate_id": str(approved[0]), "mode": "existing"}
        pending = conn.execute(
            """
            SELECT gate_id
            FROM lucidota_control.governance_gate
            WHERE workflow_id=%s AND run_id=%s AND approval_status='pending'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (workflow, run_id),
        ).fetchone()
        if not auto_approve:
            return {"status": "missing", "pending_gate_id": str(pending[0]) if pending else None}
        if pending:
            gate_id = pending[0]
            conn.execute(
                """
                UPDATE lucidota_control.governance_gate
                SET approval_status='approved', decided_at=now(),
                    evidence=evidence || %s::jsonb
                WHERE gate_id=%s
                """,
                (jdump({"signed_by": signer, "decision_note": "dispatch auto-approve test lane"}), gate_id),
            )
        else:
            gate_id = conn.execute(
                """
                INSERT INTO lucidota_control.governance_gate
                  (workflow_id, run_id, action_kind, requested_by, target, risk_level,
                   policy_mode, approval_status, rationale, evidence, decided_at)
                VALUES (%s, %s, 'internal_write', 'lucidota_dbos_dispatch',
                        %s, 'low', 'user_controlled', 'approved',
                        'dispatch auto-approve test lane', %s::jsonb, now())
                RETURNING gate_id
                """,
                (workflow, run_id, f"workflow:{workflow}", jdump({"signed_by": signer})),
            ).fetchone()[0]
        conn.commit()
    return {"status": "approved", "gate_id": str(gate_id), "mode": "auto_approved"}


def command_for(registry: dict[str, Any], argv: list[str], append_json: bool) -> list[str]:
    command = registry["command"]
    path = ROOT / command
    if path.suffix == ".py":
        cmd = [str(PY), str(path)]
    else:
        cmd = [str(path if path.exists() else command)]
    cmd.extend(argv)
    if append_json and "--json" not in cmd:
        cmd.append("--json")
    return cmd


@DBOS.step()
def dispatch_step(
    workflow: str,
    run_id: str,
    argv: list[str],
    retries: int,
    auto_approve: bool,
    signer: str,
    append_json: bool,
) -> dict[str, Any]:
    registry = registry_row(workflow)
    signoff = ensure_signoff(workflow, run_id, auto_approve, signer)
    if signoff.get("status") != "approved":
        emit_event(workflow, run_id, "dispatch", "waiting_user", {"signoff": signoff})
        raise PermissionError(f"workflow {workflow} run {run_id} needs signoff: {signoff}")

    cmd = command_for(registry, argv, append_json)
    emit_event(workflow, run_id, "dispatch", "running", {"command": cmd, "signoff": signoff})
    attempts: list[dict[str, Any]] = []
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, retries + 2):
        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        elapsed = round(time.time() - started, 3)
        last = proc
        attempts.append(
            {
                "attempt": attempt,
                "returncode": proc.returncode,
                "elapsed_s": elapsed,
                "stdout_tail": proc.stdout[-800:],
                "stderr_tail": proc.stderr[-800:],
            }
        )
        if proc.returncode == 0:
            parsed: Any
            try:
                parsed = json.loads(proc.stdout)
            except Exception:
                parsed = {"stdout": proc.stdout[-2000:]}
            event_id = emit_event(
                workflow,
                run_id,
                "dispatch",
                "succeeded",
                {"attempts": attempts, "result": parsed, "registry": registry},
            )
            return {"event_id": event_id, "workflow": workflow, "run_id": run_id, "attempts": attempts, "result": parsed}
        if attempt <= retries:
            emit_event(workflow, run_id, "dispatch_retry", "running", {"attempt": attempt, "returncode": proc.returncode})

    assert last is not None
    event_id = emit_event(
        workflow,
        run_id,
        "dispatch",
        "failed",
        {"attempts": attempts, "command": cmd, "registry": registry},
    )
    raise RuntimeError(f"workflow {workflow} failed after {len(attempts)} attempt(s); event={event_id}; stderr={last.stderr[-500:]}")


@DBOS.workflow()
def dispatch_workflow(
    workflow: str,
    run_id: str,
    argv: list[str],
    retries: int,
    auto_approve: bool,
    signer: str,
    append_json: bool,
) -> dict[str, Any]:
    emit_event(workflow, run_id, "dispatch", "queued", {"argv": argv, "retries": retries})
    return dispatch_step(workflow, run_id, argv, retries, auto_approve, signer, append_json)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-dispatch")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--signer", default="operator")
    ap.add_argument("--no-json-flag", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("workflow")
    args, workflow_args = ap.parse_known_args()
    if workflow_args and workflow_args[0] == "--":
        workflow_args = workflow_args[1:]
    run_id = args.run_id or f"dispatch-{uuid.uuid4()}"
    DBOS(config={"name": "lucidota-dbos-dispatch", "system_database_url": STATE_DB})
    DBOS.launch()
    result = dispatch_workflow(
        args.workflow,
        run_id,
        workflow_args,
        max(0, args.retries),
        args.auto_approve,
        args.signer,
        not args.no_json_flag,
    )
    DBOS.destroy(destroy_registry=True)
    report = {"ok": True, "dispatch": result}
    print(json.dumps(report, sort_keys=True, default=str) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
