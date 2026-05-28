#!/usr/bin/env python3
"""Audit graph promotion materializations for journal/command/evidence completeness."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/graph"
SCHEMAS = [ROOT / "06_SCHEMA/034_graph_promotion_pipeline.sql", ROOT / "06_SCHEMA/052_graph_promotion_materialization.sql", ROOT / "06_SCHEMA/065_graph_materialization_helper_v2.sql"]


def now() -> str:
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
    path = OUT / f"graph_journal_completeness_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                for schema in SCHEMAS:
                    cur.execute(schema.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"action": "init_schema", "execute_performed": bool(args.execute), "schemas": [rel(s) for s in SCHEMAS]})
    return 0


def check(args: argparse.Namespace) -> int:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM lucidota_go.graph_promotion_materialization")
            total = int(cur.fetchone()["n"])
            cur.execute(
                """
                SELECT m.materialization_uuid::text, m.packet_uuid::text, m.decision_uuid::text,
                       m.journal_uuid::text, m.command_envelope_uuid::text,
                       jsonb_array_length(m.evidence_refs) AS evidence_count,
                       j.journal_uuid IS NOT NULL AS journal_exists,
                       p.packet_uuid IS NOT NULL AS packet_exists,
                       d.decision_uuid IS NOT NULL AS decision_exists,
                       h.helper_receipt_uuid::text AS helper_receipt_uuid
                FROM lucidota_go.graph_promotion_materialization m
                LEFT JOIN lucidota_go.graph_journal j ON j.journal_uuid=m.journal_uuid
                LEFT JOIN lucidota_go.graph_promotion_packet p ON p.packet_uuid=m.packet_uuid
                LEFT JOIN lucidota_go.graph_promotion_decision d ON d.decision_uuid=m.decision_uuid
                LEFT JOIN lucidota_go.graph_materialization_helper_receipt h ON h.materialization_uuid=m.materialization_uuid
                ORDER BY m.created_at DESC
                """
            )
            rows = [dict(r) for r in cur.fetchall()]
    blockers = []
    warnings = []
    for row in rows:
        prefix = row["materialization_uuid"]
        if not row["journal_exists"]:
            blockers.append(f"missing_journal:{prefix}")
        if not row["packet_exists"]:
            blockers.append(f"missing_packet:{prefix}")
        if not row["decision_exists"]:
            blockers.append(f"missing_decision:{prefix}")
        if not row.get("command_envelope_uuid"):
            blockers.append(f"missing_command_envelope:{prefix}")
        if int(row.get("evidence_count") or 0) <= 0:
            blockers.append(f"missing_evidence_refs:{prefix}")
        if not row.get("helper_receipt_uuid"):
            warnings.append(f"missing_helper_receipt:{prefix}")
    passed = not blockers
    report = {
        "action": "check",
        "status": "PASS" if passed else "FAIL",
        "execute_performed": False,
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "materialization_count": total,
        "rows": rows,
        "blockers": blockers,
        "warnings": warnings,
    }
    write_report("pass" if passed else "fail", report)
    print("GRAPH_JOURNAL_COMPLETENESS=" + report["status"])
    print(f"MATERIALIZATIONS={total}")
    print(f"BLOCKERS={len(blockers)}")
    print(f"WARNINGS={len(warnings)}")
    return 0 if passed else 4


def main() -> int:
    p = argparse.ArgumentParser(description="Check graph materializations have journal and command refs")
    p.add_argument("--database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("check"); sp.set_defaults(func=check)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
