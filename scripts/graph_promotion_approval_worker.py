#!/usr/bin/env python3
"""Graph promotion approval state machine.

Enforces candidate -> defer/reject/operator_confirmed -> materialized. It writes
promotion packet/decision/transition receipts only. Materialized state requires
an existing graph_promotion_materialization row that links back to the packet.
"""
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
OUT = ROOT / "05_OUTPUTS" / "graph"
SCHEMAS = [
    ROOT / "06_SCHEMA/034_graph_promotion_pipeline.sql",
    ROOT / "06_SCHEMA/044_graph_promotion_policy_roles.sql",
    ROOT / "06_SCHEMA/052_graph_promotion_materialization.sql",
    ROOT / "06_SCHEMA/069_graph_promotion_approval_state_machine.sql",
    ROOT / "06_SCHEMA/077_graph_promotion_packet_dedupe.sql",
]
AUTH = {
    "raw_evidence",
    "operator_authored_assertion",
    "operator_defined_label",
    "deterministic_metric",
    "statistical_finding",
    "model_computed_finding",
    "stream_ml_finding",
    "graph_inferred_relation",
    "operator_confirmed_finding",
    "canonical_doctrine",
    "external_action_authorized",
}


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
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"graph_promotion_approval_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def parse_json(raw: str) -> dict[str, Any]:
    p = Path(raw)
    if len(raw) < 240 and p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON payload must be an object")
    return data


def refs(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        if not item:
            continue
        text = item.strip()
        if text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    out.extend(str(x) for x in data if str(x).strip())
                    continue
            except Exception:
                pass
        out.extend(part.strip() for part in text.split(",") if part.strip())
    return out


def as_uuid(value: str, name: str, blockers: list[str]) -> str:
    try:
        return str(uuid.UUID(value))
    except Exception:
        blockers.append(f"invalid_{name}")
        return value


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args), autocommit=True) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {
        "action": "init_schema",
        "execute_performed": bool(args.execute),
        "schemas": [rel(s) for s in SCHEMAS],
    })
    return 0


def create_candidate(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    evidence = refs(args.evidence_ref)
    if not evidence:
        blockers.append("evidence_refs_required")
    if args.authority_class not in AUTH:
        blockers.append("invalid_authority_class")
    try:
        payload = parse_json(args.candidate_payload_json)
    except Exception as exc:
        payload = {}
        blockers.append(f"invalid_candidate_payload:{exc}")
    packet_uuid = None
    if args.execute and not blockers:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_go.graph_promotion_packet(
                      source_system, candidate_kind, candidate_payload, evidence_refs, authority_class, promotion_status, detail
                    )
                    VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,'candidate',%s::jsonb)
                    ON CONFLICT (packet_dedupe_key) DO UPDATE SET
                      detail = lucidota_go.graph_promotion_packet.detail || EXCLUDED.detail,
                      created_at = lucidota_go.graph_promotion_packet.created_at
                    RETURNING packet_uuid::text, (xmax = 0) AS inserted_new
                    """,
                    (
                        args.source_system,
                        args.candidate_kind,
                        json.dumps(payload),
                        json.dumps(evidence),
                        args.authority_class,
                        json.dumps({"script": "scripts/graph_promotion_approval_worker.py", "payload_sha256": sha_obj(payload)}),
                    ),
                )
                packet_row = cur.fetchone()
                packet_uuid = packet_row["packet_uuid"]
                inserted_new = bool(packet_row["inserted_new"])
            conn.commit()
    report = {
        "action": "create_candidate",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute and packet_uuid),
        "graph_writes_performed": False,
        "packet_uuid": packet_uuid,
        "inserted_new": locals().get("inserted_new", False),
        "evidence_refs": evidence,
        "blockers": blockers,
    }
    write_report("candidate_execute" if args.execute else "candidate_dry_run", report)
    if packet_uuid:
        print(f"PACKET_UUID={packet_uuid}")
    return 0 if not blockers else 2


def decide(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    packet_uuid = as_uuid(args.packet_uuid, "packet_uuid", blockers)
    evidence = refs(args.evidence_ref)
    command_uuid = as_uuid(args.command_envelope_uuid, "command_envelope_uuid", blockers) if args.command_envelope_uuid else None
    if not evidence:
        blockers.append("evidence_refs_required")
    if args.decision not in {"defer", "reject", "operator_confirmed"}:
        blockers.append("decision_must_be_defer_reject_or_operator_confirmed")
    if args.decision == "operator_confirmed" and not command_uuid:
        blockers.append("operator_confirmed_requires_command_envelope_uuid")
    result: dict[str, Any] = {
        "action": "decide",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "packet_uuid": packet_uuid,
        "decision": args.decision,
        "decision_uuid": None,
        "transition_uuid": None,
        "from_status": None,
        "to_status": args.decision,
        "blockers": blockers,
    }
    if not blockers:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT promotion_status::text, evidence_refs FROM lucidota_go.graph_promotion_packet WHERE packet_uuid=%s::uuid FOR UPDATE", (packet_uuid,))
                row = cur.fetchone()
                if row is None:
                    blockers.append("packet_not_found")
                else:
                    result["from_status"] = row["promotion_status"]
                    if row["promotion_status"] != "candidate":
                        blockers.append(f"illegal_transition_from_{row['promotion_status']}_to_{args.decision}")
                if args.execute and not blockers:
                    cur.execute(
                        """
                        INSERT INTO lucidota_go.graph_promotion_decision(
                          packet_uuid, decision, decided_by, rationale, evidence_refs, operator_confirmed, command_envelope_uuid
                        )
                        VALUES (%s::uuid,%s,'operator',%s,%s::jsonb,%s,%s::uuid)
                        RETURNING decision_uuid::text
                        """,
                        (packet_uuid, args.decision, args.rationale, json.dumps(evidence), args.decision == "operator_confirmed", command_uuid),
                    )
                    decision_uuid = cur.fetchone()["decision_uuid"]
                    cur.execute("UPDATE lucidota_go.graph_promotion_packet SET promotion_status=%s WHERE packet_uuid=%s::uuid", (args.decision, packet_uuid))
                    cur.execute(
                        """
                        INSERT INTO lucidota_go.graph_promotion_state_transition(
                          packet_uuid, from_status, to_status, decision_uuid, command_envelope_uuid, evidence_refs, detail
                        )
                        VALUES (%s::uuid,%s,%s,%s::uuid,%s::uuid,%s::jsonb,%s::jsonb)
                        RETURNING transition_uuid::text
                        """,
                        (packet_uuid, result["from_status"], args.decision, decision_uuid, command_uuid, json.dumps(evidence), json.dumps({"script": "scripts/graph_promotion_approval_worker.py"})),
                    )
                    transition_uuid = cur.fetchone()["transition_uuid"]
                    result.update({"db_writes_performed": True, "decision_uuid": decision_uuid, "transition_uuid": transition_uuid})
                if args.execute and blockers:
                    conn.rollback()
                else:
                    conn.commit()
    result["blockers"] = blockers
    write_report("decide_execute" if args.execute else "decide_dry_run", result)
    if result["decision_uuid"]:
        print(f"DECISION_UUID={result['decision_uuid']}")
    if result["transition_uuid"]:
        print(f"TRANSITION_UUID={result['transition_uuid']}")
    return 0 if not blockers else 2


def mark_materialized(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    packet_uuid = as_uuid(args.packet_uuid, "packet_uuid", blockers)
    materialization_uuid = as_uuid(args.materialization_uuid, "materialization_uuid", blockers)
    evidence = refs(args.evidence_ref)
    if not evidence:
        blockers.append("evidence_refs_required")
    result: dict[str, Any] = {
        "action": "mark_materialized",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "packet_uuid": packet_uuid,
        "materialization_uuid": materialization_uuid,
        "transition_uuid": None,
        "from_status": None,
        "to_status": "materialized",
        "blockers": blockers,
    }
    if not blockers:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT promotion_status::text FROM lucidota_go.graph_promotion_packet WHERE packet_uuid=%s::uuid FOR UPDATE", (packet_uuid,))
                packet = cur.fetchone()
                if packet is None:
                    blockers.append("packet_not_found")
                else:
                    result["from_status"] = packet["promotion_status"]
                    if packet["promotion_status"] != "operator_confirmed":
                        blockers.append(f"illegal_transition_from_{packet['promotion_status']}_to_materialized")
                cur.execute(
                    """
                    SELECT decision_uuid::text, command_envelope_uuid::text
                    FROM lucidota_go.graph_promotion_materialization
                    WHERE materialization_uuid=%s::uuid AND packet_uuid=%s::uuid
                    """,
                    (materialization_uuid, packet_uuid),
                )
                mat = cur.fetchone()
                if mat is None:
                    blockers.append("materialization_not_found_for_packet")
                if args.execute and not blockers:
                    cur.execute("UPDATE lucidota_go.graph_promotion_packet SET promotion_status='materialized' WHERE packet_uuid=%s::uuid", (packet_uuid,))
                    cur.execute(
                        """
                        INSERT INTO lucidota_go.graph_promotion_state_transition(
                          packet_uuid, from_status, to_status, decision_uuid, materialization_uuid, command_envelope_uuid, evidence_refs, detail
                        )
                        VALUES (%s::uuid,%s,'materialized',%s::uuid,%s::uuid,%s::uuid,%s::jsonb,%s::jsonb)
                        RETURNING transition_uuid::text
                        """,
                        (packet_uuid, result["from_status"], mat["decision_uuid"], materialization_uuid, mat["command_envelope_uuid"], json.dumps(evidence), json.dumps({"script": "scripts/graph_promotion_approval_worker.py"})),
                    )
                    transition_uuid = cur.fetchone()["transition_uuid"]
                    result.update({"db_writes_performed": True, "transition_uuid": transition_uuid})
                if args.execute and blockers:
                    conn.rollback()
                else:
                    conn.commit()
    result["blockers"] = blockers
    write_report("materialized_execute" if args.execute else "materialized_dry_run", result)
    if result["transition_uuid"]:
        print(f"TRANSITION_UUID={result['transition_uuid']}")
    return 0 if not blockers else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph promotion approval state machine")
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("create-candidate")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--source-system", default="approval_worker")
    p.add_argument("--candidate-kind", choices=["node", "edge", "property", "doctrine", "workflow", "other"], default="node")
    p.add_argument("--candidate-payload-json", required=True)
    p.add_argument("--evidence-ref", action="append", default=[])
    p.add_argument("--authority-class", default="operator_authored_assertion")
    p.set_defaults(func=create_candidate)
    p = sub.add_parser("decide")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--packet-uuid", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--evidence-ref", action="append", default=[])
    p.add_argument("--command-envelope-uuid")
    p.add_argument("--rationale", default="Graph promotion approval state transition.")
    p.set_defaults(func=decide)
    p = sub.add_parser("mark-materialized")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--packet-uuid", required=True)
    p.add_argument("--materialization-uuid", required=True)
    p.add_argument("--evidence-ref", action="append", default=[])
    p.set_defaults(func=mark_materialized)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
