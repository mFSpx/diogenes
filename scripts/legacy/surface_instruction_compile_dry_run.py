#!/usr/bin/env python3
"""Dry-run compiler from generated-surface interaction to conversation instruction.

This is not button telemetry. It turns an operator selection/rejection/refinement
into a plain-language instruction plus auditable command envelope for later
conversation_command / DBOS routing. It performs no DB, graph, or canonical writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "surfaces"
DEFAULT_TEMPLATE = (
    "Tell LUCIDOTA to {action} target {target_refs} using evidence {evidence_refs} "
    "and artifact refs {artifact_refs}. Current state: {current_object_state}. "
    "Allowed effect: {allowed_effect}. Keep this as a conversation/control-plane instruction; "
    "do not mutate canonical graph or state directly."
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def split_refs(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    p = Path(raw)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("current state must be a JSON object")
    return data


def database_url(value: str | None) -> str:
    return value or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def enqueue_conversation_command(db_url: str, instruction: str, envelope: dict[str, Any], payload_sha256: str) -> tuple[str, bool]:
    import psycopg  # local dependency used by DBOS scripts
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.conversation_command
                  (plain_language_instruction, command_envelope, source_surface_id, source_artifact_refs, target_refs, evidence_refs, allowed_effect, authority_class, idempotency_key, detail)
                VALUES (%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,'operator_authored_assertion',%s,%s::jsonb)
                ON CONFLICT (cep_dedupe_key) DO UPDATE SET updated_at=lucidota_control.conversation_command.updated_at
                RETURNING command_uuid::text, (xmax = 0) AS inserted_new
            """, (
                instruction, json.dumps(envelope), envelope.get("surface_id"),
                json.dumps(envelope.get("artifact_refs", [])), json.dumps(envelope.get("target_refs", [])), json.dumps(envelope.get("evidence_refs", [])),
                envelope.get("allowed_effect", "stage_only"), payload_sha256, json.dumps({"source":"surface_instruction_compile"}),
            ))
            command_uuid, inserted_new = cur.fetchone()
        conn.commit()
    return command_uuid, bool(inserted_new)


def queue_dbos_work_order(db_url: str, command_uuid: str) -> tuple[str, str, bool]:
    import psycopg
    from psycopg.rows import dict_row
    from kernel_control_packet import dbos_enqueue_packet

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT command_uuid::text, plain_language_instruction, command_envelope,
                       source_surface_id, source_artifact_refs, target_refs, evidence_refs,
                       allowed_effect, authority_class, canonical_mutation_allowed,
                       conversation_required, status
                FROM lucidota_control.conversation_command
                WHERE command_uuid=%s::uuid
                FOR UPDATE
                """,
                (command_uuid,),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"conversation_command_not_found:{command_uuid}")
            if row["canonical_mutation_allowed"]:
                raise RuntimeError("canonical_mutation_allowed_must_be_false")
            if not row["conversation_required"]:
                raise RuntimeError("conversation_required_must_be_true")
            if row["status"] not in {"staged", "queued"}:
                raise RuntimeError(f"unsupported_conversation_command_status:{row['status']}")
            idempotency_key = f"surface_instruction_compile:{command_uuid}:dbos:v1"
            lane = "surface_instruction_compiled_command"
            source_path = str(row["source_surface_id"] or "surface_instruction_compile")
            kernel_authorization = dbos_enqueue_packet(
                queue_name="surface_cep",
                lane=lane,
                source_path=source_path,
                idempotency_key=idempotency_key,
                authorized_by="surface_instruction_compiler",
            )
            payload = {
                "command_uuid": row["command_uuid"],
                "bridge_version": "v3",
                "lane": lane,
                "source_path": source_path,
                "idempotency_key": idempotency_key,
                "kernel_authorization": kernel_authorization,
                "plain_language_instruction": row["plain_language_instruction"],
                "command_envelope": row["command_envelope"],
                "source_surface_id": row["source_surface_id"],
                "source_artifact_refs": row["source_artifact_refs"],
                "target_refs": row["target_refs"],
                "evidence_refs": row["evidence_refs"],
                "allowed_effect": row["allowed_effect"],
                "authority_class": row["authority_class"],
                "compiler": "scripts/surface_instruction_compile_dry_run.py",
            }
            cur.execute(
                """
                INSERT INTO lucidota_control.dbos_queue_job(
                  queue_name, workflow_name, job_kind, idempotency_key, payload, status, detail
                )
                VALUES ('surface_cep','surface-instruction-compiler-execute','surface_instruction_compiled_command',
                        %s,%s::jsonb,'queued',%s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET
                  updated_at=lucidota_control.dbos_queue_job.updated_at
                RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                """,
                (idempotency_key, json.dumps(payload), json.dumps({"source": "scripts/surface_instruction_compile_dry_run.py"})),
            )
            job_row = cur.fetchone()
            job_uuid = job_row["job_uuid"]
            inserted_new = bool(job_row["inserted_new"])
            cur.execute(
                """
                INSERT INTO lucidota_control.conversation_command_acceptance(
                  command_uuid, job_uuid, queue_name, idempotency_key, accepted_by, detail
                )
                VALUES (%s::uuid,%s::uuid,'surface_cep',%s,'surface_instruction_compiler',%s::jsonb)
                ON CONFLICT (command_uuid) DO UPDATE SET
                  detail=lucidota_control.conversation_command_acceptance.detail || EXCLUDED.detail
                RETURNING acceptance_uuid::text
                """,
                (
                    command_uuid,
                    job_uuid,
                    idempotency_key,
                    json.dumps({"job_inserted_new": inserted_new, "compiler_execute_worker": True}),
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
                """,
                (json.dumps({"queued_job_uuid": job_uuid, "acceptance_uuid": acceptance_uuid}), command_uuid),
            )
            if inserted_new:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.dbos_queue_event(job_uuid, queue_name, event_kind, event_source, detail)
                    VALUES (%s::uuid,'surface_cep','enqueued','surface_instruction_compiler',%s::jsonb)
                    """,
                    (job_uuid, json.dumps({"command_uuid": command_uuid, "acceptance_uuid": acceptance_uuid})),
                )
        conn.commit()
    return job_uuid, acceptance_uuid, inserted_new


def main() -> int:
    ap = argparse.ArgumentParser(description="Compile generated-surface interaction into plain-language operator instruction")
    ap.add_argument("--surface-id", required=True)
    ap.add_argument("--surface-kind", default="generated_instruction_compiler")
    ap.add_argument("--operator-action", choices=["selected", "rejected", "refined", "compared", "inspected", "operator_defined"], required=True)
    ap.add_argument("--target-ref", action="append", default=[])
    ap.add_argument("--evidence-refs")
    ap.add_argument("--artifact-refs")
    ap.add_argument("--current-object-state", default="{}", help="JSON object or path to JSON file")
    ap.add_argument("--allowed-effect", required=True)
    ap.add_argument("--instruction-template", default=DEFAULT_TEMPLATE)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true", help="write staged conversation_command row; no canonical graph/state mutation")
    ap.add_argument("--queue-dbos", action="store_true", help="after --execute, enqueue the staged conversation command into the surface_cep DBOS queue")
    ap.add_argument("--database-url")
    args = ap.parse_args()

    target_refs = args.target_ref or []
    evidence_refs = split_refs(args.evidence_refs)
    artifact_refs = split_refs(args.artifact_refs)
    state = parse_json(args.current_object_state)
    variables = {
        "action": args.operator_action,
        "target_refs": ", ".join(target_refs) or "NONE",
        "evidence_refs": ", ".join(evidence_refs) or "NONE",
        "artifact_refs": ", ".join(artifact_refs) or "NONE",
        "current_object_state": json.dumps(state, sort_keys=True, ensure_ascii=False),
        "allowed_effect": args.allowed_effect,
    }
    instruction = args.instruction_template.format(**variables)
    envelope = {
        "protocol": "lucidota.surface_instruction_envelope.v1",
        "surface_id": args.surface_id,
        "surface_kind": args.surface_kind,
        "operator_action": args.operator_action,
        "plain_language_instruction": instruction,
        "target_refs": target_refs,
        "evidence_refs": evidence_refs,
        "artifact_refs": artifact_refs,
        "current_object_state": state,
        "allowed_effect": args.allowed_effect,
        "canonical_mutation_allowed": False,
        "conversation_required": True,
        "staging_only": True,
        "route_target": "lucidota_control.conversation_command_or_dbos_workflow_envelope",
        "primary_signal": "plain_language_operator_instruction",
        "button_telemetry_primary": False,
    }
    execute = bool(args.execute)
    command_uuid = None
    inserted_new = False
    dbos_job_uuid = None
    acceptance_uuid = None
    job_inserted_new = False
    blockers: list[str] = []
    if args.queue_dbos and not execute:
        blockers.append("queue_dbos_requires_execute")
    if execute:
        try:
            db_url = database_url(args.database_url)
            command_uuid, inserted_new = enqueue_conversation_command(db_url, instruction, envelope, sha256_obj(envelope))
            if args.queue_dbos and command_uuid:
                dbos_job_uuid, acceptance_uuid, job_inserted_new = queue_dbos_work_order(db_url, command_uuid)
        except Exception as exc:
            blockers.append(f"conversation_command_insert_failed:{exc}")
    report = {
        "schema": "lucidota.surface_instruction_compile.report.v1",
        "generated_at": now_iso(),
        "mode": "execute" if execute else "dry_run",
        "surface_id": args.surface_id,
        "plain_language_instruction": instruction,
        "command_envelope": envelope,
        "payload_sha256": sha256_obj(envelope),
        "conversation_command_uuid": command_uuid,
        "inserted_new": inserted_new,
        "dbos_job_uuid": dbos_job_uuid,
        "acceptance_uuid": acceptance_uuid,
        "job_inserted_new": job_inserted_new,
        "db_writes_performed": bool(command_uuid),
        "graph_writes_performed": False,
        "canonical_mutation_allowed": False,
        "conversation_required": True,
        "blockers": blockers,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"surface_instruction_compile_{'execute' if execute else 'dry_run'}_{stamp()}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    print(f"PLAIN_LANGUAGE_INSTRUCTION={instruction}")
    if command_uuid:
        print(f"COMMAND_UUID={command_uuid}")
    if dbos_job_uuid:
        print(f"DBOS_JOB_UUID={dbos_job_uuid}")
    if acceptance_uuid:
        print(f"ACCEPTANCE_UUID={acceptance_uuid}")
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
