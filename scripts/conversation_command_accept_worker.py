#!/usr/bin/env python3
"""Accept staged conversation_command rows into ABSURD queue work orders.

Generated surfaces remain conversation instruments. This worker takes already
staged plain-language instructions, validates the no-direct-mutation contract,
queues a ABSURD work order, and advances the command status to queued.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from spine_authority_checker import decide_authority

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "absurd"
SCHEMAS = [
    ROOT / "06_SCHEMA/035_absurd_queue_spine.sql",
    ROOT / "06_SCHEMA/039_absurd_real_work_loop.sql",
    ROOT / "06_SCHEMA/068_conversation_command_acceptance.sql",
]
QUEUE = "surface_cep"
WORKFLOW = "conversation-command-acceptance"
JOB_KIND = "conversation_command_work_order"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"conversation_command_accept_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "schemas": [rel(s) for s in SCHEMAS],
    })
    return 0


def fetch_staged(cur: psycopg.Cursor[Any], limit: int, command_uuid: str | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT command_uuid::text, command_kind, plain_language_instruction, command_envelope,
               source_surface_id, source_artifact_refs, target_refs, evidence_refs, allowed_effect,
               authority_class, canonical_mutation_allowed, conversation_required, status,
               idempotency_key, detail, created_at
        FROM lucidota_control.conversation_command c
        WHERE status='staged'
          AND canonical_mutation_allowed=false
          AND conversation_required=true
          AND NOT EXISTS (
            SELECT 1 FROM lucidota_control.conversation_command_acceptance a
            WHERE a.command_uuid=c.command_uuid
          )
    """
    params: list[Any] = []
    if command_uuid:
        sql += " AND command_uuid=%s::uuid"
        params.append(command_uuid)
    sql += """
        ORDER BY created_at ASC
        LIMIT %s
        """
    params.append(limit)
    cur.execute(sql, tuple(params))
    return [dict(row) for row in cur.fetchall()]


def accept_one(cur: psycopg.Cursor[Any], command: dict[str, Any], worker_id: str) -> dict[str, Any]:
    from kernel_control_packet import absurd_enqueue_packet

    blockers: list[str] = []
    lane = "conversation_command_work_order"
    if command["canonical_mutation_allowed"]:
        blockers.append("canonical_mutation_allowed_must_be_false")
    if not command["conversation_required"]:
        blockers.append("conversation_required_must_be_true")
    if command["status"] != "staged":
        blockers.append("command_status_must_be_staged")
    evidence_refs = []
    for key in ("evidence_refs", "source_artifact_refs", "target_refs"):
        value = command.get(key)
        if isinstance(value, list):
            evidence_refs.extend(str(x) for x in value if str(x).strip())
        elif value:
            evidence_refs.append(str(value))
    authority_decision = decide_authority(
        authority_class=command.get("authority_class"),
        effect=command.get("allowed_effect"),
        lane=lane if "lane" in locals() else "conversation_command_work_order",
        evidence_refs=evidence_refs,
    )
    if not authority_decision["allowed"]:
        blockers.extend(["authority_decision:" + b for b in authority_decision["blockers"]])
    if blockers:
        return {"command_uuid": command["command_uuid"], "accepted": False, "authority_decision": authority_decision, "blockers": blockers}
    idempotency_key = f"conversation_command:{command['command_uuid']}:accept:v1"
    source_path = str(command["source_surface_id"] or "conversation_command_accept_worker")
    kernel_authorization = absurd_enqueue_packet(
        queue_name=QUEUE,
        lane=lane,
        source_path=source_path,
        idempotency_key=idempotency_key,
        authorized_by=worker_id,
    )
    payload = {
        "command_uuid": command["command_uuid"],
        "bridge_version": "v3",
        "lane": lane,
        "source_path": source_path,
        "idempotency_key": idempotency_key,
        "kernel_authorization": kernel_authorization,
        "plain_language_instruction": command["plain_language_instruction"],
        "command_envelope": command["command_envelope"],
        "source_surface_id": command["source_surface_id"],
        "source_artifact_refs": command["source_artifact_refs"],
        "target_refs": command["target_refs"],
        "evidence_refs": command["evidence_refs"],
        "allowed_effect": command["allowed_effect"],
        "authority_class": command["authority_class"],
        "accepted_by": worker_id,
        "authority_decision": authority_decision,
    }
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_job(
          queue_name, workflow_name, job_kind, idempotency_key, payload, status, detail
        )
        VALUES (%s,%s,%s,%s,%s::jsonb,'queued',%s::jsonb)
        ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET
          updated_at=lucidota_control.absurd_queue_job.updated_at
        RETURNING job_uuid::text, (xmax = 0) AS inserted_new
        """,
        (
            QUEUE,
            WORKFLOW,
            JOB_KIND,
            idempotency_key,
            json.dumps(payload),
            json.dumps({"source": "scripts/conversation_command_accept_worker.py"}),
        ),
    )
    job_row = cur.fetchone()
    job_uuid = job_row["job_uuid"]
    inserted_new = job_row["inserted_new"]
    cur.execute(
        """
        INSERT INTO lucidota_control.conversation_command_acceptance(
          command_uuid, job_uuid, queue_name, idempotency_key, accepted_by, detail
        )
        VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb)
        ON CONFLICT (command_uuid) DO UPDATE SET
          detail=lucidota_control.conversation_command_acceptance.detail || EXCLUDED.detail
        RETURNING acceptance_uuid::text
        """,
        (
            command["command_uuid"],
            job_uuid,
            QUEUE,
            idempotency_key,
            worker_id,
            json.dumps({"job_inserted_new": bool(inserted_new)}),
        ),
    )
    acceptance_uuid = cur.fetchone()["acceptance_uuid"]
    cur.execute(
        """
        UPDATE lucidota_control.conversation_command
        SET status='queued',
            accepted_at=coalesce(accepted_at, now()),
            updated_at=now(),
            detail=detail || %s::jsonb
        WHERE command_uuid=%s::uuid
          AND status='staged'
        """,
        (json.dumps({"queued_job_uuid": job_uuid, "acceptance_uuid": acceptance_uuid}), command["command_uuid"]),
    )
    if inserted_new:
        cur.execute(
            """
            INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail)
            VALUES (%s::uuid,%s,'enqueued','conversation_command_accept_worker',%s::jsonb)
            """,
            (job_uuid, QUEUE, json.dumps({"command_uuid": command["command_uuid"], "acceptance_uuid": acceptance_uuid})),
        )
    return {
        "command_uuid": command["command_uuid"],
        "job_uuid": job_uuid,
        "acceptance_uuid": acceptance_uuid,
        "job_inserted_new": bool(inserted_new),
        "accepted": True,
        "authority_decision": authority_decision,
        "blockers": [],
    }


def accept(args: argparse.Namespace) -> int:
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    accepted: list[dict[str, Any]] = []
    blockers: list[str] = []
    with psycopg.connect(state_db(args)) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            commands = fetch_staged(cur, args.limit, args.command_uuid)
            if args.execute:
                for command in commands:
                    result = accept_one(cur, command, worker_id)
                    accepted.append(result)
                    blockers.extend(result.get("blockers", []))
                conn.commit()
        if not args.execute:
            accepted = [{"command_uuid": c["command_uuid"], "would_queue": True, "allowed_effect": c["allowed_effect"]} for c in commands]
    report = {
        "action": "accept",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "worker_id": worker_id,
        "commands_seen": len(accepted),
        "commands_accepted": sum(1 for item in accepted if item.get("accepted")),
        "jobs_queued": sum(1 for item in accepted if item.get("job_uuid")),
        "accepted": accepted,
        "blockers": blockers if blockers else ([] if accepted else ["no_staged_conversation_commands"]),
    }
    write_report("execute" if args.execute else "dry_run", report)
    print(f"COMMANDS_SEEN={len(accepted)}")
    print(f"JOBS_QUEUED={report['jobs_queued'] if args.execute else 0}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Accept staged conversation commands into ABSURD queue jobs")
    parser.add_argument("--state-database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("accept")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--worker-id")
    p.add_argument("--command-uuid")
    p.set_defaults(func=accept)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
