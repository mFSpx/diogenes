#!/usr/bin/env python3
"""Materialize an evidence-gated graph promotion packet into canonical graph.

This is the promotion path, not a direct graph-write shortcut:
- verifies a conversation_command UUID exists in the control database
- runs graph_promotion_preflight with materialize_requested=true
- sets SET LOCAL lucidota.graph_promotion_path='on' inside one transaction
- writes graph_item + graph_journal + materialization receipt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from graph_materialization_helper import command_materialization_policy

SCHEMAS = [
    ROOT / "06_SCHEMA/016_go_graph_core.sql",
    ROOT / "06_SCHEMA/034_graph_promotion_pipeline.sql",
    ROOT / "06_SCHEMA/040_graph_write_barrier_enforcement.sql",
    ROOT / "06_SCHEMA/044_graph_promotion_policy_roles.sql",
    ROOT / "06_SCHEMA/052_graph_promotion_materialization.sql",
    ROOT / "06_SCHEMA/114_balanced_ternary_valency.sql",
]
OUT = ROOT / "05_OUTPUTS/graph"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"graph_promotion_materialize_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def control_db(args: argparse.Namespace) -> str:
    return args.control_database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def parse_payload(raw: str) -> dict[str, Any]:
    stripped = raw.lstrip()
    if not stripped.startswith(("{", "[")) and len(raw) < 240:
        p = Path(raw)
    else:
        p = None
    if p is not None and p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("candidate payload must be a JSON object")
    return data


def split_refs(raw: list[str]) -> list[str]:
    refs: list[str] = []
    for item in raw:
        if not item:
            continue
        if item.strip().startswith("["):
            try:
                refs.extend(str(x) for x in json.loads(item))
                continue
            except Exception:
                pass
        refs.extend(part.strip() for part in item.split(",") if part.strip())
    return refs


def ternary_valency(payload: dict[str, Any]) -> int:
    value = payload.get("ternary_valency")
    if value is None and isinstance(payload.get("payload"), dict):
        value = payload["payload"].get("ternary_valency")
    try:
        val = int(value)
    except (TypeError, ValueError):
        return 0
    return val if val in {-1, 0, 1} else 0


def command_exists(db_url: str, command_uuid: str) -> bool:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('lucidota_control.conversation_command')")
            if cur.fetchone()[0] is None:
                return False
            cur.execute("SELECT 1 FROM lucidota_control.conversation_command WHERE command_uuid=%s::uuid", (command_uuid,))
            return cur.fetchone() is not None


def counts(cur: psycopg.Cursor[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in ("graph_item", "graph_edge", "graph_journal", "graph_promotion_materialization"):
        cur.execute(f"SELECT count(*) AS n FROM lucidota_go.{table}")
        row = cur.fetchone()
        out[table] = int(row["n"] if isinstance(row, dict) else row[0])
    return out


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(storage_db(args)) as conn:
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


def materialize(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    try:
        command_uuid = str(uuid.UUID(args.command_envelope_uuid))
    except Exception:
        command_uuid = args.command_envelope_uuid
        blockers.append("invalid_command_envelope_uuid")
    try:
        payload = parse_payload(args.candidate_payload_json)
    except Exception as exc:
        payload = {}
        blockers.append(f"invalid_candidate_payload:{type(exc).__name__}:{exc}")
    evidence_refs = split_refs(args.evidence_ref)
    if not evidence_refs:
        blockers.append("evidence_refs_required")
    command_policy: dict[str, Any] = {}
    if not blockers:
        command_policy = command_materialization_policy(control_db(args), command_uuid)
        if not command_policy["allowed"]:
            blockers.extend(f"command_policy:{b}" for b in command_policy["blockers"])
    if args.execute and os.environ.get("LUCIDOTA_GRAPH_MATERIALIZATION_HELPER") != "scripts/graph_materialization_helper.py":
        blockers.append("materialization_helper_required")

    result: dict[str, Any] = {
        "action": "materialize",
        "execute_performed": False,
        "db_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "promotion_path_used": False,
        "command_envelope_uuid": command_uuid,
        "evidence_refs": evidence_refs,
        "command_policy": command_policy,
        "candidate_payload_sha256": sha256_obj(payload),
        "preflight": None,
        "counts_before": None,
        "counts_after": None,
        "packet_uuid": None,
        "decision_uuid": None,
        "graph_item_uuid": None,
        "journal_uuid": None,
        "materialization_uuid": None,
        "blockers": blockers,
    }
    if blockers:
        write_report("blocked", result)
        return 2

    with psycopg.connect(storage_db(args)) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            result["counts_before"] = counts(cur)
            cur.execute(
                "SELECT lucidota_go.graph_promotion_preflight(%s,%s,%s,%s,%s::jsonb,%s::uuid) AS preflight",
                ("graph_promoter", "operator_confirmed", True, True, json.dumps(evidence_refs), command_uuid),
            )
            preflight = cur.fetchone()["preflight"]
            result["preflight"] = preflight
            if not preflight.get("allowed"):
                blockers.extend(preflight.get("blockers", []))
                result["blockers"] = blockers
                conn.rollback()
                write_report("preflight_blocked", result)
                return 2
            if not args.execute:
                conn.rollback()
                write_report("dry_run", result)
                return 0
            term = str(payload.get("term") or "CLAIM")
            label = str(payload.get("label") or payload.get("claim") or "Promoted candidate")[:300]
            status = str(payload.get("status") or "staged")
            if status not in {"located", "staged"}:
                status = "staged"
            cur.execute("SELECT 1 FROM lucidota_go.term_registry WHERE term=%s", (term,))
            if cur.fetchone() is None:
                blockers.append(f"term_not_in_registry:{term}")
                result["blockers"] = blockers
                conn.rollback()
                write_report("preflight_blocked", result)
                return 2
            graph_payload = dict(payload)
            valency = ternary_valency(payload)
            graph_payload["ternary_valency"] = valency
            graph_payload.setdefault("evidence_note", "Materialized through graph promotion path with command envelope.")
            graph_payload.setdefault("promotion_payload_sha256", sha256_obj(payload))
            cur.execute(
                """
                INSERT INTO lucidota_go.graph_promotion_packet
                  (source_system, candidate_kind, candidate_payload, evidence_refs, authority_class, promotion_status, detail)
                VALUES (%s, 'node', %s::jsonb, %s::jsonb, %s, 'operator_confirmed', %s::jsonb)
                RETURNING packet_uuid::text
                """,
                (
                    args.source_system,
                    json.dumps(payload),
                    json.dumps(evidence_refs),
                    args.authority_class,
                    json.dumps({"script": "scripts/graph_promotion_materialize.py", "command_envelope_uuid": command_uuid}),
                ),
            )
            packet_uuid = cur.fetchone()["packet_uuid"]
            cur.execute(
                """
                INSERT INTO lucidota_go.graph_promotion_decision
                  (packet_uuid, decision, decided_by, rationale, evidence_refs, operator_confirmed, command_envelope_uuid)
                VALUES (%s::uuid, 'operator_confirmed', 'operator', %s, %s::jsonb, true, %s::uuid)
                RETURNING decision_uuid::text
                """,
                (packet_uuid, args.rationale, json.dumps(evidence_refs), command_uuid),
            )
            decision_uuid = cur.fetchone()["decision_uuid"]
            cur.execute("SET LOCAL lucidota.graph_promotion_path='on'")
            if os.environ.get("LUCIDOTA_GRAPH_MATERIALIZATION_HELPER") == "scripts/graph_materialization_helper.py":
                cur.execute("SET LOCAL lucidota.graph_materialization_helper='scripts/graph_materialization_helper.py'")
                result["materialization_helper_path"] = "scripts/graph_materialization_helper.py"
            else:
                result["materialization_helper_path"] = None
            result["promotion_path_used"] = True
            cur.execute(
                """
                INSERT INTO lucidota_go.graph_item(term, label, status, location_at_on_graph, location_real_at_added, ternary_valency, payload)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
                RETURNING uuid::text
                """,
                (
                    term,
                    label,
                    status,
                    "graph_promotion_materialize.py",
                    json.dumps({"command_envelope_uuid": command_uuid, "evidence_refs": evidence_refs}),
                    valency,
                    json.dumps(graph_payload),
                ),
            )
            graph_item_uuid = cur.fetchone()["uuid"]
            cur.execute(
                """
                INSERT INTO lucidota_go.graph_journal(item_uuid, action, reason, after_state)
                VALUES (%s::uuid, 'stage', %s, %s::jsonb)
                RETURNING journal_uuid::text
                """,
                (
                    graph_item_uuid,
                    args.rationale,
                    json.dumps({"packet_uuid": packet_uuid, "decision_uuid": decision_uuid, "command_envelope_uuid": command_uuid, "graph_item_uuid": graph_item_uuid}),
                ),
            )
            journal_uuid = cur.fetchone()["journal_uuid"]
            cur.execute(
                """
                INSERT INTO lucidota_go.graph_promotion_materialization(
                  packet_uuid, decision_uuid, graph_item_uuid, journal_uuid, command_envelope_uuid,
                  evidence_refs, materialization_kind, detail
                )
                VALUES (%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::jsonb,'node',%s::jsonb)
                RETURNING materialization_uuid::text
                """,
                (
                    packet_uuid,
                    decision_uuid,
                    graph_item_uuid,
                    journal_uuid,
                    command_uuid,
                    json.dumps(evidence_refs),
                    json.dumps({"script": "scripts/graph_promotion_materialize.py"}),
                ),
            )
            materialization_uuid = cur.fetchone()["materialization_uuid"]
            result.update({
                "execute_performed": True,
                "db_writes_performed": True,
                "canonical_graph_writes_performed": True,
                "packet_uuid": packet_uuid,
                "decision_uuid": decision_uuid,
                "graph_item_uuid": graph_item_uuid,
                "journal_uuid": journal_uuid,
                "materialization_uuid": materialization_uuid,
            })
            result["counts_after"] = counts(cur)
        conn.commit()
    write_report("execute", result)
    print(f"PACKET_UUID={result['packet_uuid']}")
    print(f"DECISION_UUID={result['decision_uuid']}")
    print(f"GRAPH_ITEM_UUID={result['graph_item_uuid']}")
    print(f"JOURNAL_UUID={result['journal_uuid']}")
    print(f"MATERIALIZATION_UUID={result['materialization_uuid']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize graph promotion packets through the guarded promotion path")
    parser.add_argument("--storage-database-url")
    parser.add_argument("--control-database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("materialize")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--command-envelope-uuid", required=True)
    p.add_argument("--candidate-payload-json", required=True)
    p.add_argument("--evidence-ref", action="append", default=[])
    p.add_argument("--source-system", default="operator_cli")
    p.add_argument("--authority-class", default="operator_authored_assertion")
    p.add_argument("--rationale", default="Operator-confirmed graph promotion materialization through command envelope.")
    p.set_defaults(func=materialize)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
