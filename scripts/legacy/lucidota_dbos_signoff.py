#!/usr/bin/env python3
"""DBOS-backed workflow signoff lane.

Purpose: make workflow approval testable in one command surface.
- request: create a governance_gate + waiting_user workflow_event
- approve/deny: decide the gate + terminal workflow_event
- smoke: request and approve a registered workflow end-to-end
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import psycopg
from dbos import DBOS, DBOSConfig

ROOT = Path(__file__).resolve().parents[1]
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
REGISTRY_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"


def jdump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def apply_registry(conn: psycopg.Connection) -> None:
    conn.execute(REGISTRY_SCHEMA.read_text())


def workflow_row(conn: psycopg.Connection, workflow_name: str) -> dict[str, Any]:
    apply_registry(conn)
    row = conn.execute(
        """
        SELECT workflow_name, owner, phase, status, command, inputs, outputs, notes
        FROM lucidota_control.workflow_registry
        WHERE workflow_name=%s
        """,
        (workflow_name,),
    ).fetchone()
    if not row:
        raise ValueError(f"unknown workflow: {workflow_name}")
    keys = ["workflow_name", "owner", "phase", "status", "command", "inputs", "outputs", "notes"]
    return dict(zip(keys, row))


def emit_event(
    conn: psycopg.Connection,
    workflow_id: str,
    run_id: str,
    phase: str,
    status: str,
    detail: dict[str, Any],
) -> str:
    event_id = conn.execute(
        """
        INSERT INTO lucidota_control.workflow_event
          (workflow_id, run_id, phase, status, source, detail)
        VALUES (%s, %s, %s, %s, 'lucidota_dbos_signoff', %s::jsonb)
        RETURNING event_id
        """,
        (workflow_id, run_id, phase, status, jdump(detail)),
    ).fetchone()[0]
    return str(event_id)


@DBOS.step()
def request_signoff_step(
    workflow_name: str,
    run_id: str,
    target: str,
    risk: str,
    requested_by: str,
    rationale: str,
) -> dict[str, Any]:
    with psycopg.connect(STATE_DB) as conn:
        wf = workflow_row(conn, workflow_name)
        gate_id = conn.execute(
            """
            INSERT INTO lucidota_control.governance_gate
              (workflow_id, run_id, action_kind, requested_by, target, risk_level,
               policy_mode, approval_status, rationale, evidence)
            VALUES (%s, %s, 'internal_write', %s, %s, %s,
                    'user_controlled', 'pending', %s, %s::jsonb)
            RETURNING gate_id
            """,
            (
                workflow_name,
                run_id,
                requested_by,
                target,
                risk,
                rationale,
                jdump({"signoff_kind": "workflow", "workflow_registry": wf}),
            ),
        ).fetchone()[0]
        event_id = emit_event(
            conn,
            workflow_name,
            run_id,
            "signoff",
            "waiting_user",
            {"gate_id": str(gate_id), "target": target, "risk": risk, "rationale": rationale},
        )
        conn.commit()
    return {"workflow": workflow_name, "run_id": run_id, "gate_id": str(gate_id), "event_id": event_id}


@DBOS.step()
def decide_signoff_step(gate_id: str, decision: str, signer: str, note: str) -> dict[str, Any]:
    if decision not in {"approved", "denied"}:
        raise ValueError("decision must be approved or denied")
    event_status = "succeeded" if decision == "approved" else "cancelled"
    with psycopg.connect(STATE_DB) as conn:
        row = conn.execute(
            """
            UPDATE lucidota_control.governance_gate
            SET approval_status=%s,
                decided_at=now(),
                evidence=evidence || %s::jsonb
            WHERE gate_id=%s AND approval_status='pending'
            RETURNING workflow_id, run_id, target, risk_level, rationale
            """,
            (
                decision,
                jdump({"signed_by": signer, "decision_note": note, "decision": decision}),
                gate_id,
            ),
        ).fetchone()
        if not row:
            existing = conn.execute(
                """
                SELECT workflow_id, run_id, approval_status
                FROM lucidota_control.governance_gate
                WHERE gate_id=%s
                """,
                (gate_id,),
            ).fetchone()
            if existing:
                raise ValueError(f"gate {gate_id} is already {existing[2]}")
            raise ValueError(f"unknown gate: {gate_id}")
        workflow_id, run_id, target, risk_level, rationale = row
        event_id = emit_event(
            conn,
            workflow_id,
            run_id,
            "signoff",
            event_status,
            {
                "gate_id": gate_id,
                "decision": decision,
                "signed_by": signer,
                "target": target,
                "risk": risk_level,
                "rationale": rationale,
                "note": note,
            },
        )
        conn.commit()
    return {
        "gate_id": gate_id,
        "workflow": workflow_id,
        "run_id": run_id,
        "decision": decision,
        "event_id": event_id,
    }


@DBOS.workflow()
def request_signoff_workflow(
    workflow_name: str,
    run_id: str,
    target: str,
    risk: str,
    requested_by: str,
    rationale: str,
) -> dict[str, Any]:
    return request_signoff_step(workflow_name, run_id, target, risk, requested_by, rationale)


@DBOS.workflow()
def decide_signoff_workflow(gate_id: str, decision: str, signer: str, note: str) -> dict[str, Any]:
    return decide_signoff_step(gate_id, decision, signer, note)


@DBOS.workflow()
def smoke_signoff_workflow(workflow_name: str, run_id: str, signer: str) -> dict[str, Any]:
    req = request_signoff_step(
        workflow_name,
        run_id,
        target=f"workflow:{workflow_name}",
        risk="low",
        requested_by="signoff-smoke",
        rationale="one-hour DBOS signoff lane smoke",
    )
    dec = decide_signoff_step(req["gate_id"], "approved", signer, "smoke approval")
    return {"request": req, "decision": dec}


def list_gates(pending_only: bool, limit: int) -> list[dict[str, Any]]:
    where = "WHERE approval_status='pending'" if pending_only else ""
    with psycopg.connect(STATE_DB) as conn:
        rows = conn.execute(
            f"""
            SELECT gate_id, workflow_id, run_id, approval_status, risk_level,
                   requested_by, target, rationale, created_at, decided_at
            FROM lucidota_control.governance_gate
            {where}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    keys = [
        "gate_id",
        "workflow",
        "run_id",
        "status",
        "risk",
        "requested_by",
        "target",
        "rationale",
        "created_at",
        "decided_at",
    ]
    return [dict(zip(keys, row)) for row in rows]


def workflow_status(workflow: str | None, run_id: str | None, limit: int) -> dict[str, Any]:
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
        events = conn.execute(
            f"""
            SELECT workflow_id, run_id, phase, status, source, detail, created_at
            FROM lucidota_control.workflow_event
            {where}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
    keys = ["workflow", "run_id", "phase", "status", "source", "detail", "created_at"]
    return {"events": [dict(zip(keys, row)) for row in events]}


def render_gates(gates: list[dict[str, Any]]) -> str:
    if not gates:
        return "No gates."
    return "\n".join(
        f"{g['status']:>8} {g['risk']:<8} {g['workflow']} run={g['run_id']} gate={g['gate_id']} target={g['target']}"
        for g in gates
    )


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-dbos-signoff")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    req = sub.add_parser("request")
    req.add_argument("--json", action="store_true")
    req.add_argument("workflow")
    req.add_argument("--run-id", default=None)
    req.add_argument("--target", default=None)
    req.add_argument("--risk", choices=["low", "medium", "high", "critical"], default="medium")
    req.add_argument("--requested-by", default="operator")
    req.add_argument("--rationale", default="operator requested workflow signoff")

    dec = sub.add_parser("approve")
    dec.add_argument("--json", action="store_true")
    dec.add_argument("gate_id")
    dec.add_argument("--signer", default="operator")
    dec.add_argument("--note", default="approved for test")

    deny = sub.add_parser("deny")
    deny.add_argument("--json", action="store_true")
    deny.add_argument("gate_id")
    deny.add_argument("--signer", default="operator")
    deny.add_argument("--note", default="denied")

    ls = sub.add_parser("list")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--pending", action="store_true")
    ls.add_argument("--limit", type=int, default=20)

    st = sub.add_parser("status")
    st.add_argument("--json", action="store_true")
    st.add_argument("--workflow")
    st.add_argument("--run-id")
    st.add_argument("--limit", type=int, default=20)

    sm = sub.add_parser("smoke")
    sm.add_argument("--json", action="store_true")
    sm.add_argument("--workflow", default="dbos-smoke")
    sm.add_argument("--run-id", default=None)
    sm.add_argument("--signer", default="operator")

    args = ap.parse_args()
    config: DBOSConfig = {"name": "lucidota-dbos-signoff", "system_database_url": STATE_DB}
    report: dict[str, Any]

    if args.cmd in {"request", "approve", "deny", "smoke"}:
        DBOS(config=config)
        DBOS.launch()
        if args.cmd == "request":
            run_id = args.run_id or f"signoff-{uuid.uuid4()}"
            report = {
                "ok": True,
                "signoff": request_signoff_workflow(
                    args.workflow,
                    run_id,
                    args.target or f"workflow:{args.workflow}",
                    args.risk,
                    args.requested_by,
                    args.rationale,
                ),
            }
        elif args.cmd == "approve":
            report = {"ok": True, "signoff": decide_signoff_workflow(args.gate_id, "approved", args.signer, args.note)}
        elif args.cmd == "deny":
            report = {"ok": True, "signoff": decide_signoff_workflow(args.gate_id, "denied", args.signer, args.note)}
        else:
            run_id = args.run_id or f"signoff-smoke-{uuid.uuid4()}"
            report = {"ok": True, "signoff_smoke": smoke_signoff_workflow(args.workflow, run_id, args.signer)}
        DBOS.destroy(destroy_registry=True)
    elif args.cmd == "list":
        gates = list_gates(args.pending, args.limit)
        report = {"ok": True, "gates": gates}
        if not args.json:
            print(render_gates(gates))
            return 0
    else:
        report = {"ok": True, "status": workflow_status(args.workflow, args.run_id, args.limit)}

    print(json.dumps(report, sort_keys=True, default=str) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
