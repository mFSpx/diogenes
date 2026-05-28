#!/usr/bin/env python3
"""Resolve graph-promotion evidence refs before packet creation.

Resolved evidence is a precondition for governed graph promotion. This resolver
checks files and known DB-backed refs and can append resolution receipts; it does
not create promotion packets and does not mutate canonical graph items/edges.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA/085_graph_promotion_evidence_resolver.sql"
OUT = ROOT / "05_OUTPUTS/graph"

DB_REF_TABLES = {
    "design_atom": ("storage", "lucidota_archaeology.design_atom", "atom_uuid"),
    "workflow_blueprint": ("storage", "lucidota_archaeology.workflow_blueprint", "blueprint_uuid"),
    "document_claim_packet": ("storage", "lucidota_korpus.document_claim_packet", "claim_packet_uuid"),
    "claim_packet": ("storage", "lucidota_korpus.document_claim_packet", "claim_packet_uuid"),
    "temporal_claim": ("storage", "lucidota_korpus.temporal_claim", "claim_uuid"),
    "graph_promotion_packet": ("storage", "lucidota_go.graph_promotion_packet", "packet_uuid"),
    "conversation_command": ("state", "lucidota_control.conversation_command", "command_uuid"),
    "command": ("state", "lucidota_control.conversation_command", "command_uuid"),
    "dbos_job": ("state", "lucidota_control.dbos_queue_job", "job_uuid"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"graph_promotion_evidence_resolver_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def split_ref(raw: str) -> tuple[str, str]:
    if ":" in raw and not re.match(r"^[A-Za-z]:[/\\]", raw):
        k, v = raw.split(":", 1)
        if k in DB_REF_TABLES or k in {"file", "path"}:
            return k, v
    return "file", raw


def db_exists(url: str, table: str, col: str, value: str) -> tuple[bool, str | None]:
    try:
        uid = str(uuid.UUID(value))
    except Exception:
        return False, "invalid_uuid"
    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) AS reg", (table,))
            if cur.fetchone()["reg"] is None:
                return False, "table_missing"
            cur.execute(f"SELECT 1 FROM {table} WHERE {col}=%s::uuid LIMIT 1", (uid,))
            return cur.fetchone() is not None, None


def resolve_one(raw: str, args: argparse.Namespace) -> dict[str, Any]:
    kind, value = split_ref(raw.strip())
    result: dict[str, Any] = {"evidence_ref": raw, "ref_kind": kind, "resolved": False, "source_table": None, "source_uuid": None, "source_path": None, "blocker": None}
    if kind in {"file", "path"}:
        p = Path(value)
        if not p.is_absolute():
            p = ROOT / p
        result["source_path"] = rel(p)
        result["resolved"] = p.exists() and p.is_file()
        if not result["resolved"]:
            result["blocker"] = "file_not_found"
        return result
    db_name, table, col = DB_REF_TABLES[kind]
    url = storage_db(args) if db_name == "storage" else state_db(args)
    ok, blocker = db_exists(url, table, col, value)
    result.update({"resolved": ok, "source_table": table, "source_uuid": value if ok else None, "blocker": blocker})
    return result


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(storage_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"action": "init_schema", "execute_performed": bool(args.execute), "schema": rel(SCHEMA)})
    return 0


def resolve(args: argparse.Namespace) -> int:
    refs = []
    for raw in args.evidence_ref:
        refs.extend([x.strip() for x in raw.split(",") if x.strip()])
    results = [resolve_one(r, args) for r in refs]
    blockers = [f"{r['evidence_ref']}:{r['blocker']}" for r in results if not r["resolved"]]
    if args.execute:
        with psycopg.connect(storage_db(args)) as conn:
            with conn.cursor() as cur:
                for r in results:
                    cur.execute(
                        """
                        INSERT INTO lucidota_go.graph_promotion_evidence_resolution(evidence_ref, ref_kind, resolved, source_table, source_uuid, source_path, detail)
                        VALUES (%s,%s,%s,%s,%s::uuid,%s,%s::jsonb)
                        """,
                        (r["evidence_ref"], r["ref_kind"], r["resolved"], r["source_table"], r["source_uuid"], r["source_path"], json.dumps({"blocker": r.get("blocker")})),
                    )
            conn.commit()
    report = {"action": "resolve", "execute_performed": bool(args.execute), "db_writes_performed": bool(args.execute), "graph_writes_performed": False, "refs_checked": len(results), "results": results, "blockers": blockers, "status": "PASS" if not blockers else "FAIL"}
    write_report("resolve_pass" if not blockers else "resolve_fail", report)
    print("EVIDENCE_RESOLUTION=" + report["status"])
    return 0 if not blockers else 4


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve graph-promotion evidence references")
    ap.add_argument("--storage-database-url")
    ap.add_argument("--state-database-url")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init-schema"); s.add_argument("--execute", action="store_true")
    r = sub.add_parser("resolve"); r.add_argument("--evidence-ref", action="append", required=True); r.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.cmd == "init-schema": return init_schema(args)
    if args.cmd == "resolve": return resolve(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
