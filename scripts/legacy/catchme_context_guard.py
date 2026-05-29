#!/usr/bin/env python3
"""Runtime CatchMe consent/sensitivity guard for context use."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA0 = ROOT / "06_SCHEMA/046_catchme_sensitivity_map.sql"
SCHEMA = ROOT / "06_SCHEMA/055_catchme_context_guard.sql"
OUT = ROOT / "05_OUTPUTS/catchme"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"catchme_context_guard_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA0.read_text(encoding="utf-8"))
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"execute_performed": bool(args.execute), "schema": rel(SCHEMA)})
    return 0


def classify_path(args: argparse.Namespace, path_ref: str) -> dict[str, Any]:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT rule_key,path_glob,sensitivity_class,consent_status,allowed_use,priority FROM lucidota_control.catchme_sensitivity_rule WHERE active ORDER BY priority ASC")
            rules = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT scope_key,sensitivity_class,consent_status,allowed_use FROM lucidota_control.catchme_scope")
            scopes = {r["scope_key"]: dict(r) for r in cur.fetchall()}
    matched = None
    for r in rules:
        if fnmatch.fnmatch(path_ref, r["path_glob"]):
            matched = r
            break
    if args.scope_key and args.scope_key in scopes:
        s = scopes[args.scope_key]
        base = {"matched_rule_key": matched["rule_key"] if matched else None, "scope_key": args.scope_key, **s}
    elif matched:
        base = {"matched_rule_key": matched["rule_key"], "scope_key": "", "sensitivity_class": matched["sensitivity_class"], "consent_status": matched["consent_status"], "allowed_use": matched["allowed_use"]}
    else:
        base = {"matched_rule_key": None, "scope_key": args.scope_key or "", "sensitivity_class": "private", "consent_status": "requires_operator", "allowed_use": "operator_review_only"}
    return base


def check(args: argparse.Namespace) -> int:
    path_ref = rel(args.path)
    c = classify_path(args, path_ref)
    allowed = False
    reason = ""
    if c["consent_status"] == "revoked" or c["sensitivity_class"] == "revoked":
        reason = "blocked:revoked"
    elif c["sensitivity_class"] == "secret" and not args.operator_approved:
        reason = "blocked:secret_requires_operator_approval"
    elif c["consent_status"] == "requires_operator" and not args.operator_approved:
        reason = "blocked:operator_approval_required"
    elif args.requested_use not in str(c.get("allowed_use", "")) and c.get("allowed_use") not in {"recall", "context_use"} and not args.operator_approved:
        reason = f"blocked:requested_use_not_in_allowed_use:{c.get('allowed_use')}"
    else:
        allowed = True
        reason = "allowed"
    report = {
        "action": "check",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "path_ref": path_ref,
        "path_sha256": sha(path_ref),
        "requested_use": args.requested_use,
        "operator_approved": bool(args.operator_approved),
        "classification": c,
        "allowed": allowed,
        "reason": reason,
        "blockers": [] if allowed else [reason],
    }
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.catchme_context_decision(
                      path_sha256,path_ref,scope_key,requested_use,sensitivity_class,consent_status,operator_approved,allowed,reason,detail
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING decision_uuid::text
                    """,
                    (report["path_sha256"], path_ref, c.get("scope_key") or "", args.requested_use, c["sensitivity_class"], c["consent_status"], bool(args.operator_approved), allowed, reason, json.dumps({"script": "scripts/catchme_context_guard.py", "matched_rule_key": c.get("matched_rule_key")})),
                )
                report["decision_uuid"] = cur.fetchone()[0]
            conn.commit()
        report["db_writes_performed"] = True
    write_report("check_execute" if args.execute else "check_dry_run", report)
    if "decision_uuid" in report:
        print(f"DECISION_UUID={report['decision_uuid']}")
    print("CATCHME_ALLOWED=" + ("true" if allowed else "false"))
    return 0 if allowed else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Check CatchMe context use against consent/sensitivity policy")
    p.add_argument("--database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("check")
    sp.add_argument("--execute", action="store_true")
    sp.add_argument("--path", required=True)
    sp.add_argument("--scope-key", default="")
    sp.add_argument("--requested-use", default="context_use")
    sp.add_argument("--operator-approved", action="store_true")
    sp.set_defaults(func=check)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
