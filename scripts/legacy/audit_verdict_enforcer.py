#!/usr/bin/env python3
"""Strict audit verdict contract enforcer."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA/056_audit_verdict_runtime_enforcement.sql"
SCHEMA0 = ROOT / "06_SCHEMA/039_dbos_real_work_loop.sql"
OUT = ROOT / "05_OUTPUTS/audit"


def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def db(args: argparse.Namespace) -> str: return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"audit_verdict_enforcer_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def refs(raw: list[str]) -> list[str]:
    out: list[str] = []
    for r in raw:
        if r.strip().startswith("["):
            try: out.extend(str(x) for x in json.loads(r)); continue
            except Exception: pass
        out.extend(x.strip() for x in r.split(",") if x.strip())
    return out


def validate(verdict: str, evidence_refs: list[str], remediation: str) -> list[str]:
    errors: list[str] = []
    if verdict not in {"PASS", "FAIL", "PARTIAL_FAIL"}:
        errors.append("invalid_verdict")
    if not evidence_refs:
        errors.append("evidence_refs_required")
    if verdict in {"FAIL", "PARTIAL_FAIL"} and not remediation.strip():
        errors.append("remediation_required_for_fail_or_partial_fail")
    return errors


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA0.read_text(encoding="utf-8"))
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"execute_performed": bool(args.execute), "schema": rel(SCHEMA)})
    return 0


def enforce(args: argparse.Namespace) -> int:
    evidence_refs = refs(args.evidence_ref)
    errors = validate(args.verdict, evidence_refs, args.remediation)
    report = {
        "action": "enforce",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "task_id": args.task_id,
        "verdict": args.verdict,
        "valid": not errors,
        "errors": errors,
        "evidence_refs": evidence_refs,
        "remediation": args.remediation,
    }
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                verdict_uuid = None
                if not errors:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.audit_verdict_contract(task_id, verdict, required_fields_ok, remediation, evidence_refs, rejected, detail)
                        VALUES (%s,%s,true,%s,%s::jsonb,false,%s::jsonb)
                        RETURNING verdict_uuid::text
                        """,
                        (args.task_id, args.verdict, args.remediation, json.dumps(evidence_refs), json.dumps({"script": "scripts/audit_verdict_enforcer.py"})),
                    )
                    verdict_uuid = cur.fetchone()[0]
                    report["verdict_uuid"] = verdict_uuid
                cur.execute(
                    """
                    INSERT INTO lucidota_control.audit_verdict_validation_event(verdict_uuid,task_id,verdict,valid,errors,evidence_refs,remediation,detail)
                    VALUES (%s::uuid,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb)
                    RETURNING validation_uuid::text
                    """,
                    (verdict_uuid, args.task_id, args.verdict, not errors, json.dumps(errors), json.dumps(evidence_refs), args.remediation, json.dumps({"script": "scripts/audit_verdict_enforcer.py"})),
                )
                report["validation_uuid"] = cur.fetchone()[0]
            conn.commit()
        report["db_writes_performed"] = True
    write_report("execute" if args.execute else "dry_run", report)
    if "validation_uuid" in report: print(f"VALIDATION_UUID={report['validation_uuid']}")
    print("AUDIT_VERDICT_VALID=" + ("true" if not errors else "false"))
    return 0 if not errors else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Enforce audit verdict contract")
    p.add_argument("--database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("enforce"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--task-id", required=True); sp.add_argument("--verdict", choices=["PASS","FAIL","PARTIAL_FAIL"], required=True); sp.add_argument("--evidence-ref", action="append", default=[]); sp.add_argument("--remediation", default=""); sp.set_defaults(func=enforce)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
