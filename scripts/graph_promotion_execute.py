#!/usr/bin/env python3
"""Execute-gated graph promotion packet/decision writer.

This is the DB execute path for the promotion pipeline only. It never writes
canonical graph_item, graph_edge, or graph_journal rows.
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

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "graph"
AUTHORITY_CLASSES = {
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
DECISIONS = {"defer", "reject", "promote", "operator_confirmed", "superseded"}
CANDIDATE_KINDS = {"node", "edge", "property", "doctrine", "workflow", "other"}
CANONICAL_TABLES = [
    "lucidota_go.graph_item",
    "lucidota_go.graph_edge",
    "lucidota_go.graph_journal",
]
PROMOTION_TABLES = [
    "lucidota_go.graph_promotion_packet",
    "lucidota_go.graph_promotion_decision",
    "lucidota_go.graph_promotion_journal_requirement",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def redact_database_url(url: str | None) -> str:
    if not url:
        return "unset"
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def connect(database_url: str):
    import psycopg  # type: ignore
    return psycopg.connect(database_url)


def table_counts(cur, tables: list[str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for table in tables:
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            out[table] = None
        else:
            cur.execute(f"SELECT count(*) FROM {table}")
            out[table] = int(cur.fetchone()[0])
    return out


def validate_args(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    try:
        payload = json.loads(args.candidate_payload_json)
    except Exception as exc:
        payload = {}
        blockers.append(f"candidate_payload_json_invalid:{exc}")
    evidence_refs = [e for e in args.evidence_ref if e.strip()]
    if not evidence_refs:
        blockers.append("evidence_ref_required")
    if args.authority_class not in AUTHORITY_CLASSES:
        blockers.append("invalid_authority_class")
    if args.candidate_kind not in CANDIDATE_KINDS:
        blockers.append("invalid_candidate_kind")
    if args.decision not in DECISIONS:
        blockers.append("invalid_decision")
    if args.decision in {"promote", "operator_confirmed"} and not args.operator_confirmed:
        blockers.append("promote_requires_operator_confirmed")
    if args.operator_confirmed and not args.command_envelope_uuid:
        blockers.append("operator_confirmed_requires_command_envelope_uuid")
    if args.command_envelope_uuid:
        try:
            uuid.UUID(args.command_envelope_uuid)
        except Exception:
            blockers.append("command_envelope_uuid_invalid")
    return payload, blockers


def run_db(args: argparse.Namespace, payload: dict[str, Any], execute: bool) -> dict[str, Any]:
    database_url = args.database_url or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"
    result: dict[str, Any] = {
        "database_url": redact_database_url(database_url),
        "schema_reachable": False,
        "promotion_tables_before": {},
        "promotion_tables_after": {},
        "canonical_counts_before": {},
        "canonical_counts_after": {},
        "packet_uuid": None,
        "decision_uuid": None,
        "db_writes_performed": False,
        "promotion_db_writes_performed": False,
        "canonical_graph_writes_performed": False,
    }
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            result["promotion_tables_before"] = table_counts(cur, PROMOTION_TABLES)
            result["canonical_counts_before"] = table_counts(cur, CANONICAL_TABLES)
            missing = [t for t, c in result["promotion_tables_before"].items() if c is None]
            result["schema_reachable"] = not missing
            if missing:
                result["missing_tables"] = missing
                return result
            if execute:
                cur.execute(
                    """
                    INSERT INTO lucidota_go.graph_promotion_packet
                      (source_system, candidate_kind, candidate_payload, evidence_refs, authority_class, detail)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                    RETURNING packet_uuid
                    """,
                    (
                        args.source_system,
                        args.candidate_kind,
                        json.dumps(payload),
                        json.dumps(args.evidence_ref),
                        args.authority_class,
                        json.dumps({"payload_sha256": sha256_obj(payload), "script": "scripts/graph_promotion_execute.py"}),
                    ),
                )
                packet_uuid = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO lucidota_go.graph_promotion_decision
                      (packet_uuid, decision, decided_by, rationale, evidence_refs, operator_confirmed, command_envelope_uuid)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                    RETURNING decision_uuid
                    """,
                    (
                        packet_uuid,
                        args.decision,
                        args.decided_by,
                        args.rationale,
                        json.dumps(args.evidence_ref),
                        bool(args.operator_confirmed),
                        args.command_envelope_uuid,
                    ),
                )
                decision_uuid = cur.fetchone()[0]
                cur.execute(
                    "UPDATE lucidota_go.graph_promotion_packet SET promotion_status=%s WHERE packet_uuid=%s",
                    (args.decision, packet_uuid),
                )
                result["packet_uuid"] = str(packet_uuid)
                result["decision_uuid"] = str(decision_uuid)
                result["db_writes_performed"] = True
                result["promotion_db_writes_performed"] = True
            result["promotion_tables_after"] = table_counts(cur, PROMOTION_TABLES)
            result["canonical_counts_after"] = table_counts(cur, CANONICAL_TABLES)
        if execute:
            conn.commit()
        else:
            conn.rollback()
    result["canonical_graph_writes_performed"] = result["canonical_counts_before"] != result["canonical_counts_after"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute-gated graph promotion packet/decision writer")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default; validate and report only")
    mode.add_argument("--execute", action="store_true", help="insert promotion packet/decision only; no canonical graph mutation")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", "postgresql:///lucidota_storage"))
    parser.add_argument("--source-system", default="operator_cli")
    parser.add_argument("--candidate-kind", default="doctrine")
    parser.add_argument("--candidate-payload-json", default='{"claim":"Graph promotion execute path smoke candidate; no canonical graph mutation."}')
    parser.add_argument("--evidence-ref", action="append", default=[])
    parser.add_argument("--authority-class", default="operator_authored_assertion")
    parser.add_argument("--decision", default="defer")
    parser.add_argument("--decided-by", default="operator", choices=["operator", "workflow", "master_eye", "graph_promoter", "other"])
    parser.add_argument("--rationale", default="Execute-path smoke: write promotion packet and defer decision only; no canonical graph mutation.")
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--command-envelope-uuid")
    args = parser.parse_args()

    execute = bool(args.execute)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload, blockers = validate_args(args)
    db_result: dict[str, Any] | None = None
    if not blockers:
        try:
            db_result = run_db(args, payload, execute)
            if not db_result.get("schema_reachable"):
                blockers.append("graph_promotion_schema_not_applied:" + ",".join(db_result.get("missing_tables", [])))
            if db_result.get("canonical_graph_writes_performed"):
                blockers.append("canonical_graph_counts_changed")
        except Exception as exc:
            blockers.append(f"db_error:{exc}")
    report = {
        "generated_at": utc_now(),
        "mode": "execute" if execute else "dry_run",
        "source_system": args.source_system,
        "candidate_kind": args.candidate_kind,
        "candidate_payload_sha256": sha256_obj(payload),
        "evidence_refs": args.evidence_ref,
        "authority_class": args.authority_class,
        "decision": args.decision,
        "operator_confirmed": bool(args.operator_confirmed),
        "db_result": db_result,
        "execute_performed": bool(execute and db_result and db_result.get("promotion_db_writes_performed") and not db_result.get("canonical_graph_writes_performed")),
        "db_writes_performed": bool(db_result and db_result.get("db_writes_performed")),
        "promotion_db_writes_performed": bool(db_result and db_result.get("promotion_db_writes_performed")),
        "canonical_graph_writes_performed": bool(db_result and db_result.get("canonical_graph_writes_performed")),
        "blockers": blockers,
    }
    out = OUT_DIR / f"graph_promotion_execute_{'execute' if execute else 'dry_run'}_{stamp()}.json"
    report["report_path"] = str(out.relative_to(ROOT))
    out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    if report["execute_performed"] and db_result:
        print(f"PACKET_UUID={db_result.get('packet_uuid')}")
        print(f"DECISION_UUID={db_result.get('decision_uuid')}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
