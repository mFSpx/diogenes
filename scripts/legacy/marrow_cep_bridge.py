#!/usr/bin/env python3
"""Bridge a Marrow Loop receipt into a staged conversation_command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "marrow_loop"
SCHEMAS = [
    ROOT / "06_SCHEMA/039_dbos_real_work_loop.sql",
    ROOT / "06_SCHEMA/072_marrow_cep_bridge.sql",
]


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


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"marrow_cep_bridge_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def receipt_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_receipt(raw: str) -> tuple[Path, dict[str, Any], str, list[str]]:
    blockers: list[str] = []
    path = receipt_path(raw)
    if not path.exists():
        return path, {}, "", ["receipt_missing"]
    data = path.read_bytes()
    try:
        receipt = json.loads(data.decode("utf-8"))
    except Exception as exc:
        return path, {}, sha_bytes(data), [f"receipt_json_invalid:{exc}"]
    if not isinstance(receipt, dict):
        blockers.append("receipt_must_be_object")
    if receipt.get("schema") != "lucidota.marrow_loop.command_receipt.v0":
        blockers.append("unsupported_receipt_schema")
    try:
        uuid.UUID(str(receipt.get("command_uuid")))
    except Exception:
        blockers.append("invalid_marrow_command_uuid")
    for key in ("raw_command", "normalized_intent", "authority_class", "source"):
        if not str(receipt.get(key, "")).strip():
            blockers.append(f"{key}_required")
    if receipt.get("db_writes_performed") is True:
        blockers.append("marrow_receipt_already_db_writing_not_bridge_safe")
    if receipt.get("graph_writes_performed") is True:
        blockers.append("marrow_receipt_graph_write_forbidden")
    return path, receipt, sha_bytes(data), blockers


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
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


def bridge(args: argparse.Namespace) -> int:
    path, receipt, receipt_sha, blockers = load_receipt(args.receipt)
    command_uuid = None
    bridge_uuid = None
    idempotency_key = ""
    envelope: dict[str, Any] = {}
    instruction = ""
    if not blockers:
        idempotency_key = f"marrow_cep:{receipt['command_uuid']}:v1"
        instruction = (
            f"{receipt['raw_command']} "
            f"(intent: {receipt['normalized_intent']}; source: {receipt['source']}; "
            f"authority: {receipt['authority_class']})."
        )
        envelope = {
            "protocol": "lucidota.marrow_cep_bridge.v1",
            "marrow_command_uuid": receipt["command_uuid"],
            "receipt_path": rel(path),
            "receipt_sha256": receipt_sha,
            "raw_command": receipt["raw_command"],
            "normalized_intent": receipt["normalized_intent"],
            "authority_class": receipt["authority_class"],
            "source": receipt["source"],
            "canonical_mutation_allowed": False,
            "conversation_required": True,
            "staging_only": True,
            "allowed_effect": args.allowed_effect,
        }
    if args.execute and not blockers:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.conversation_command(
                      command_kind, plain_language_instruction, command_envelope,
                      source_surface_id, source_artifact_refs, target_refs, evidence_refs,
                      allowed_effect, authority_class, idempotency_key, detail
                    )
                    VALUES ('marrow_receipt',%s,%s::jsonb,'marrow_loop',%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (idempotency_key) DO UPDATE SET
                      updated_at=lucidota_control.conversation_command.updated_at
                    RETURNING command_uuid::text
                    """,
                    (
                        instruction,
                        json.dumps(envelope),
                        json.dumps([rel(path)]),
                        json.dumps(args.target_ref or []),
                        json.dumps([rel(path), f"marrow_command_uuid:{receipt['command_uuid']}"]),
                        args.allowed_effect,
                        receipt["authority_class"],
                        idempotency_key,
                        json.dumps({"source": "scripts/marrow_cep_bridge.py"}),
                    ),
                )
                command_uuid = cur.fetchone()["command_uuid"]
                cur.execute(
                    """
                    INSERT INTO lucidota_control.marrow_cep_bridge_receipt(
                      marrow_command_uuid, conversation_command_uuid, receipt_path, receipt_sha256,
                      idempotency_key, detail
                    )
                    VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (marrow_command_uuid) DO UPDATE SET
                      detail=lucidota_control.marrow_cep_bridge_receipt.detail || EXCLUDED.detail
                    RETURNING bridge_uuid::text
                    """,
                    (
                        receipt["command_uuid"],
                        command_uuid,
                        rel(path),
                        receipt_sha,
                        idempotency_key,
                        json.dumps({"conversation_command_uuid": command_uuid}),
                    ),
                )
                bridge_uuid = cur.fetchone()["bridge_uuid"]
            conn.commit()
    report = {
        "action": "bridge",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute and command_uuid),
        "graph_writes_performed": False,
        "receipt_path": rel(path),
        "receipt_sha256": receipt_sha,
        "marrow_command_uuid": receipt.get("command_uuid"),
        "conversation_command_uuid": command_uuid,
        "bridge_uuid": bridge_uuid,
        "idempotency_key": idempotency_key,
        "plain_language_instruction": instruction,
        "command_envelope": envelope,
        "blockers": blockers,
    }
    write_report("execute" if args.execute else "dry_run", report)
    if command_uuid:
        print(f"CONVERSATION_COMMAND_UUID={command_uuid}")
    if bridge_uuid:
        print(f"BRIDGE_UUID={bridge_uuid}")
    return 0 if not blockers else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge Marrow receipt into staged conversation_command")
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("bridge")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--receipt", required=True)
    p.add_argument("--target-ref", action="append", default=[])
    p.add_argument("--allowed-effect", default="stage_conversation_command")
    p.set_defaults(func=bridge)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
