#!/usr/bin/env python3
"""Reusable hardening helper for graph promotion materialization.

This helper does not provide a direct graph-write shortcut. It wraps the
existing guarded promotion path and then proves the required receipt chain:

conversation_command -> promotion packet -> decision -> graph_item/edge ->
graph_journal -> graph_promotion_materialization -> helper receipt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.runtime_dsns import resolve_state_dsn, resolve_storage_dsn
OUT = ROOT / "05_OUTPUTS" / "graph"
SCHEMAS = [
    ROOT / "06_SCHEMA/016_go_graph_core.sql",
    ROOT / "06_SCHEMA/034_graph_promotion_pipeline.sql",
    ROOT / "06_SCHEMA/040_graph_write_barrier_enforcement.sql",
    ROOT / "06_SCHEMA/044_graph_promotion_policy_roles.sql",
    ROOT / "06_SCHEMA/052_graph_promotion_materialization.sql",
    ROOT / "06_SCHEMA/065_graph_materialization_helper_v2.sql",
]
MATERIALIZER = ROOT / "scripts/graph_promotion_materialize.py"
MATERIALIZATION_COMMAND_STATUSES = {"queued", "accepted", "executed"}
MATERIALIZATION_AUTHORITY_CLASSES = {
    "operator_authored_assertion",
    "operator_confirmed_finding",
    "canonical_doctrine",
    "external_action_authorized",
}
MATERIALIZATION_POLICY_VALUES = {
    "graph_promoter_transaction",
    "canonical_graph_materialization_via_graph_promoter",
    "graph_materialization_helper",
}
NEGATIVE_GRAPH_MATERIALIZATION_RE = re.compile(
    r"\b(?:no|without|disable(?:d)?|forbid(?:den)?|refuse(?:d)?)\b.{0,80}\bgraph materialization\b"
    r"|\bgraph materialization\b.{0,80}\b(?:not allowed|disabled|forbidden|refused)\b",
    re.I,
)
POSITIVE_GRAPH_MATERIALIZATION_RE = re.compile(
    r"\b(?:canonical graph materialization|graph materialization|materiali[sz]e canonical graph)\b",
    re.I,
)


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


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or resolve_storage_dsn()


def control_db(args: argparse.Namespace) -> str:
    return args.control_database_url or resolve_state_dsn()


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"graph_materialization_helper_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def split_refs(raw: list[str]) -> list[str]:
    refs: list[str] = []
    for item in raw:
        if not item:
            continue
        text = item.strip()
        if text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    refs.extend(str(x) for x in data if str(x).strip())
                    continue
            except Exception:
                pass
        refs.extend(part.strip() for part in text.split(",") if part.strip())
    return refs


def valid_uuid(value: str) -> str:
    return str(uuid.UUID(value))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return [str(x) for x in data if str(x).strip()]
            except Exception:
                pass
        return [text]
    return [str(value)]


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def fetch_command(db_url: str, command_uuid: str) -> dict[str, Any] | None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT to_regclass('lucidota_control.conversation_command')")
            regclass_row = cur.fetchone()
            if regclass_row is None or next(iter(regclass_row.values())) is None:
                return None
            cur.execute(
                """
                SELECT command_uuid::text, status, allowed_effect, authority_class,
                       canonical_mutation_allowed, conversation_required,
                       evidence_refs, target_refs, command_envelope
                FROM lucidota_control.conversation_command
                WHERE command_uuid=%s::uuid
                """,
                (command_uuid,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def command_exists(db_url: str, command_uuid: str) -> bool:
    return fetch_command(db_url, command_uuid) is not None


def evaluate_command_materialization_policy(command: dict[str, Any] | None) -> dict[str, Any]:
    """Return whether a conversation_command authorizes graph materialization.

    A command being present is not enough. The command must explicitly authorize
    graph materialization through the graph-promoter transaction while still
    preserving the conversation-command no-direct-mutation contract.
    """
    blockers: list[str] = []
    if not command:
        return {"allowed": False, "blockers": ["command_envelope_uuid_not_found_in_control_db"]}

    envelope = as_dict(command.get("command_envelope"))
    allowed_effect = str(command.get("allowed_effect") or envelope.get("allowed_effect") or "")
    envelope_effect = str(envelope.get("allowed_effect") or "")
    combined_effect = " ".join(part for part in [allowed_effect, envelope_effect] if part).strip()
    evidence_refs = as_list(command.get("evidence_refs")) or as_list(envelope.get("evidence_refs"))
    target_refs = as_list(command.get("target_refs")) or as_list(envelope.get("target_refs"))
    policy = str(envelope.get("graph_materialization_policy") or envelope.get("materialization_policy") or "")

    if command.get("status") not in MATERIALIZATION_COMMAND_STATUSES:
        blockers.append("command_status_must_be_queued_accepted_or_executed")
    if NEGATIVE_GRAPH_MATERIALIZATION_RE.search(combined_effect):
        blockers.append("allowed_effect_explicitly_forbids_graph_materialization")
    if not POSITIVE_GRAPH_MATERIALIZATION_RE.search(combined_effect):
        blockers.append("allowed_effect_must_explicitly_authorize_graph_materialization")
    if policy not in MATERIALIZATION_POLICY_VALUES:
        blockers.append("command_envelope_graph_materialization_policy_required")
    if envelope.get("staging_only") is True:
        blockers.append("command_envelope_staging_only_forbids_materialization")
    if command.get("canonical_mutation_allowed") is not False:
        blockers.append("conversation_command_direct_canonical_mutation_flag_must_remain_false")
    if command.get("conversation_required") is not True:
        blockers.append("conversation_required_must_be_true")
    if command.get("authority_class") not in MATERIALIZATION_AUTHORITY_CLASSES:
        blockers.append("authority_class_not_allowed_for_materialization")
    if not evidence_refs:
        blockers.append("command_evidence_refs_required")
    if not target_refs:
        blockers.append("command_target_refs_required")

    return {
        "allowed": not blockers,
        "blockers": blockers,
        "status": command.get("status"),
        "allowed_effect": allowed_effect,
        "authority_class": command.get("authority_class"),
        "evidence_count": len(evidence_refs),
        "target_refs": target_refs,
        "graph_materialization_policy": policy,
    }


def command_materialization_policy(db_url: str, command_uuid: str) -> dict[str, Any]:
    return evaluate_command_materialization_policy(fetch_command(db_url, command_uuid))


def parse_report_path(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            p = ROOT / line.split("=", 1)[1].strip()
            if p.exists():
                return p
    return None


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


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


def load_materialization(conn: psycopg.Connection[Any], materialization_uuid: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
              m.materialization_uuid::text,
              m.packet_uuid::text,
              m.decision_uuid::text,
              m.graph_item_uuid::text,
              m.graph_edge_uuid::text,
              m.journal_uuid::text,
              m.command_envelope_uuid::text,
              m.evidence_refs,
              jsonb_array_length(m.evidence_refs) AS materialization_evidence_count,
              m.materialization_kind,
              p.authority_class,
              p.promotion_status::text,
              p.evidence_refs AS packet_evidence_refs,
              jsonb_array_length(p.evidence_refs) AS packet_evidence_count,
              d.decision::text,
              d.operator_confirmed,
              d.command_envelope_uuid::text AS decision_command_envelope_uuid,
              d.evidence_refs AS decision_evidence_refs,
              jsonb_array_length(d.evidence_refs) AS decision_evidence_count,
              j.action,
              j.after_state
            FROM lucidota_go.graph_promotion_materialization m
            JOIN lucidota_go.graph_promotion_packet p ON p.packet_uuid = m.packet_uuid
            JOIN lucidota_go.graph_promotion_decision d ON d.decision_uuid = m.decision_uuid
            JOIN lucidota_go.graph_journal j ON j.journal_uuid = m.journal_uuid
            WHERE m.materialization_uuid=%s::uuid
            """,
            (materialization_uuid,),
        )
        return cur.fetchone()


def verify_materialization(
    storage_url: str,
    control_url: str,
    materialization_uuid: str,
    expected_command_uuid: str | None,
    expected_evidence_refs: list[str],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    with psycopg.connect(storage_url) as conn:
        row = load_materialization(conn, materialization_uuid)
    if row is None:
        return {}, ["materialization_uuid_not_found"]

    command_uuid = str(row["command_envelope_uuid"])
    if expected_command_uuid and command_uuid != expected_command_uuid:
        blockers.append("command_uuid_mismatch")
    if not command_exists(control_url, command_uuid):
        blockers.append("command_envelope_uuid_not_found_in_control_db")
    if int(row["materialization_evidence_count"]) <= 0:
        blockers.append("materialization_evidence_refs_required")
    if int(row["packet_evidence_count"]) <= 0:
        blockers.append("packet_evidence_refs_required")
    if int(row["decision_evidence_count"]) <= 0:
        blockers.append("decision_evidence_refs_required")
    if expected_evidence_refs:
        have = {str(x) for x in row["evidence_refs"]}
        missing = [x for x in expected_evidence_refs if x not in have]
        if missing:
            blockers.append(f"missing_expected_evidence_refs:{','.join(missing)}")
    if row["decision"] != "operator_confirmed" or not row["operator_confirmed"]:
        blockers.append("operator_confirmed_decision_required")
    if row["promotion_status"] != "operator_confirmed":
        blockers.append("packet_operator_confirmed_status_required")
    if row["decision_command_envelope_uuid"] != command_uuid:
        blockers.append("decision_command_uuid_mismatch")
    if row["action"] not in {"stage", "promote", "materialize"}:
        blockers.append("journal_action_not_materialization_related")
    policy = command_materialization_policy(control_url, command_uuid)
    if not policy["allowed"]:
        blockers.extend(f"command_policy:{b}" for b in policy["blockers"])
    if not row["journal_uuid"]:
        blockers.append("journal_receipt_required")
    if row["materialization_kind"] == "node" and not row["graph_item_uuid"]:
        blockers.append("node_materialization_requires_graph_item_uuid")
    if row["materialization_kind"] == "edge" and not row["graph_edge_uuid"]:
        blockers.append("edge_materialization_requires_graph_edge_uuid")
    return dict(row), blockers


def insert_helper_receipt(
    storage_url: str,
    row: dict[str, Any],
    materializer_report_path: str,
    detail: dict[str, Any],
) -> str:
    with psycopg.connect(storage_url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO lucidota_go.graph_materialization_helper_receipt(
                  materialization_uuid, packet_uuid, decision_uuid, graph_item_uuid, graph_edge_uuid,
                  journal_uuid, command_envelope_uuid, evidence_count, authority_class,
                  verification_passed, materializer_report_path, detail
                )
                VALUES (%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s,%s,true,%s,%s::jsonb)
                ON CONFLICT (materialization_uuid) DO UPDATE SET
                  materializer_report_path=EXCLUDED.materializer_report_path,
                  detail=lucidota_go.graph_materialization_helper_receipt.detail || EXCLUDED.detail
                RETURNING helper_receipt_uuid::text
                """,
                (
                    row["materialization_uuid"],
                    row["packet_uuid"],
                    row["decision_uuid"],
                    row.get("graph_item_uuid"),
                    row.get("graph_edge_uuid"),
                    row["journal_uuid"],
                    row["command_envelope_uuid"],
                    int(row["materialization_evidence_count"]),
                    row["authority_class"],
                    materializer_report_path,
                    json.dumps(detail),
                ),
            )
            receipt_uuid = cur.fetchone()["helper_receipt_uuid"]
        conn.commit()
    return receipt_uuid


def verify(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    try:
        materialization_uuid = valid_uuid(args.materialization_uuid)
    except Exception:
        materialization_uuid = args.materialization_uuid
        blockers.append("invalid_materialization_uuid")
    expected_command_uuid = None
    if args.command_envelope_uuid:
        try:
            expected_command_uuid = valid_uuid(args.command_envelope_uuid)
        except Exception:
            expected_command_uuid = args.command_envelope_uuid
            blockers.append("invalid_command_envelope_uuid")
    evidence_refs = split_refs(args.evidence_ref)
    row: dict[str, Any] = {}
    if not blockers:
        row, blockers = verify_materialization(storage_db(args), control_db(args), materialization_uuid, expected_command_uuid, evidence_refs)
    report = {
        "action": "verify",
        "execute_performed": False,
        "db_writes_performed": False,
        "materialization_uuid": materialization_uuid,
        "expected_command_envelope_uuid": expected_command_uuid,
        "expected_evidence_refs": evidence_refs,
        "verification_passed": not blockers,
        "materialization": row,
        "blockers": blockers,
    }
    write_report("verify_pass" if not blockers else "verify_blocked", report)
    return 0 if not blockers else 2


def materialize(args: argparse.Namespace) -> int:
    blockers: list[str] = []
    try:
        command_uuid = valid_uuid(args.command_envelope_uuid)
    except Exception:
        command_uuid = args.command_envelope_uuid
        blockers.append("invalid_command_envelope_uuid")
    evidence_refs = split_refs(args.evidence_ref)
    if not evidence_refs:
        blockers.append("evidence_refs_required")
    if args.execute and not getattr(args, "operator_confirmed", False):
        blockers.append("operator_confirmation_required")
    command_policy: dict[str, Any] = {}
    if not blockers:
        command_policy = command_materialization_policy(control_db(args), command_uuid)
        if not command_policy["allowed"]:
            blockers.extend(f"command_policy:{b}" for b in command_policy["blockers"])
    if blockers:
        write_report("materialize_blocked", {
            "action": "materialize",
            "execute_performed": False,
            "db_writes_performed": False,
            "command_envelope_uuid": command_uuid,
            "evidence_refs": evidence_refs,
            "command_policy": command_policy,
            "blockers": blockers,
        })
        return 2

    materializer_script = Path(args.materializer_script).resolve() if args.materializer_script else MATERIALIZER
    if not materializer_script.exists():
        write_report("materialize_blocked", {
            "action": "materialize",
            "execute_performed": False,
            "db_writes_performed": False,
            "command_envelope_uuid": command_uuid,
            "evidence_refs": evidence_refs,
            "blockers": ["materializer_script_missing"],
            "materializer_script": rel(materializer_script),
        })
        return 2
    cmd = [
        sys.executable,
        str(materializer_script),
        "--storage-database-url",
        storage_db(args),
        "--control-database-url",
        control_db(args),
        "materialize",
        "--command-envelope-uuid",
        command_uuid,
        "--candidate-payload-json",
        args.candidate_payload_json,
        "--source-system",
        args.source_system,
        "--authority-class",
        args.authority_class,
        "--rationale",
        args.rationale,
    ]
    for ref in evidence_refs:
        cmd.extend(["--evidence-ref", ref])
    if args.execute:
        cmd.append("--execute")
    env = os.environ.copy()
    env["LUCIDOTA_GRAPH_MATERIALIZATION_HELPER"] = "scripts/graph_materialization_helper.py"
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    materializer_report = parse_report_path(proc.stdout)
    materializer_payload: dict[str, Any] = {}
    if materializer_report:
        materializer_payload = read_json(materializer_report)
    else:
        blockers.append("materializer_report_path_missing")
    if proc.returncode != 0:
        blockers.append(f"materializer_rc_{proc.returncode}")

    db_result = materializer_payload.get("db_result") if isinstance(materializer_payload.get("db_result"), dict) else {}
    materialization_uuid = materializer_payload.get("materialization_uuid") or db_result.get("materialization_uuid")
    packet_uuid = materializer_payload.get("packet_uuid") or db_result.get("packet_uuid")
    decision_uuid = materializer_payload.get("decision_uuid") or db_result.get("decision_uuid")
    verification: dict[str, Any] = {}
    helper_receipt_uuid = None
    if not blockers and args.execute:
        if materialization_uuid:
            verification, verify_blockers = verify_materialization(storage_db(args), control_db(args), str(materialization_uuid), command_uuid, evidence_refs)
            blockers.extend(verify_blockers)
            if not blockers:
                helper_receipt_uuid = insert_helper_receipt(
                    storage_db(args),
                    verification,
                    rel(materializer_report or ""),
                    {
                        "script": "scripts/graph_materialization_helper.py",
                        "materializer_stdout": proc.stdout[-4000:],
                        "materializer_stderr": proc.stderr[-4000:],
                    },
                )
        elif not args.promotion_only:
            blockers.append("materialization_uuid_missing")

    expected_deltas = {"graph_item": 1, "graph_edge": 0, "graph_journal": 1, "graph_promotion_materialization": 1} if args.execute else {"graph_item": 0, "graph_edge": 0, "graph_journal": 0, "graph_promotion_materialization": 0}
    counts_before = materializer_payload.get("counts_before") or {}
    counts_after = materializer_payload.get("counts_after") or {}
    actual_deltas = {k: (int(counts_after.get(k, counts_before.get(k, 0))) - int(counts_before.get(k, 0))) for k in set(counts_before) | set(counts_after)} if counts_before else {}
    if args.execute and not blockers and actual_deltas:
        for key, expected in expected_deltas.items():
            if actual_deltas.get(key) != expected:
                blockers.append(f"unexpected_delta:{key}:{actual_deltas.get(key)}!={expected}")
    report = {
        "action": "materialize",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute and not blockers),
        "canonical_graph_writes_performed": bool(materializer_payload.get("canonical_graph_writes_performed")),
        "counts_before": counts_before,
        "counts_after": counts_after,
        "expected_deltas": expected_deltas,
        "actual_deltas": actual_deltas,
        "promotion_path_used": bool(materializer_payload.get("promotion_path_used")),
        "command_envelope_uuid": command_uuid,
        "evidence_refs": evidence_refs,
        "command_policy": command_policy,
        "materializer_returncode": proc.returncode,
        "materializer_report_path": rel(materializer_report) if materializer_report else None,
        "materialization_uuid": materialization_uuid,
        "packet_uuid": packet_uuid,
        "decision_uuid": decision_uuid,
        "graph_item_uuid": materializer_payload.get("graph_item_uuid"),
        "journal_uuid": materializer_payload.get("journal_uuid"),
        "helper_receipt_uuid": helper_receipt_uuid,
        "verification_passed": bool(args.execute and ((helper_receipt_uuid is not None) or args.promotion_only) and not blockers) or bool((not args.execute) and not blockers),
        "verification": verification,
        "promotion_only": bool(args.promotion_only),
        "materializer_script": rel(materializer_script),
        "blockers": blockers,
        "materializer_stdout_tail": proc.stdout[-4000:],
        "materializer_stderr_tail": proc.stderr[-4000:],
    }
    write_report("materialize_execute" if args.execute else "materialize_dry_run", report)
    if materialization_uuid:
        print(f"MATERIALIZATION_UUID={materialization_uuid}")
    if helper_receipt_uuid:
        print(f"HELPER_RECEIPT_UUID={helper_receipt_uuid}")
    return 0 if not blockers else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Require command envelope + evidence + journal receipt for graph materialization")
    parser.add_argument("--storage-database-url")
    parser.add_argument("--control-database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)

    p = sub.add_parser("materialize")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--operator-confirmed", action="store_true")
    p.add_argument("--command-envelope-uuid", required=True)
    p.add_argument("--candidate-payload-json", required=True)
    p.add_argument("--evidence-ref", action="append", default=[])
    p.add_argument("--source-system", default="graph_materialization_helper")
    p.add_argument("--authority-class", default="operator_authored_assertion")
    p.add_argument("--rationale", default="Operator-confirmed graph promotion materialization through reusable helper.")
    p.add_argument("--promotion-only", action="store_true", help="allow packet+decision promotion verification when no materialization UUID is emitted")
    p.add_argument("--materializer-script", default=str(MATERIALIZER), help="materializer command script path")
    p.set_defaults(func=materialize)

    p = sub.add_parser("verify")
    p.add_argument("--materialization-uuid", required=True)
    p.add_argument("--command-envelope-uuid")
    p.add_argument("--evidence-ref", action="append", default=[])
    p.set_defaults(func=verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
