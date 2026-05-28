#!/usr/bin/env python3
"""Executable DeMem runtime boundary guard.

Given a plain-language instruction, checks active lucidota_control.demem_boundary
rules and returns allow/warn/rewrite/block. Execute mode records the decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA0 = ROOT / "06_SCHEMA/039_dbos_real_work_loop.sql"
SCHEMA = ROOT / "06_SCHEMA/054_demem_runtime_enforcement.sql"
OUT = ROOT / "05_OUTPUTS/demem"

PATTERNS = {
    "generated_not_policy_mutable": re.compile(r"\b(generated|draft|surface|artifact).{0,80}\b(policy|canonical|mutable|approved|execute)\b", re.I),
    "retrieved_not_verified": re.compile(r"\b(retrieved|search result|candidate|recall).{0,80}\b(verified|truth|fact|answer)\b", re.I),
    "repeated_not_preferred": re.compile(r"\b(repeated|often|usually|habit).{0,80}\b(preferred|preference|want|should)\b", re.I),
    "surface_not_ui": re.compile(r"\b(surface|dashboard|button|toggle).{0,80}\b(UI|mutate|write|canonical|state)\b", re.I),
    "graph_path_not_evidence": re.compile(r"\b(graph path|graph walk|associative trail|hipporag).{0,80}\b(evidence|proof|truth)\b", re.I),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"demem_runtime_guard_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
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


def active_boundaries(args: argparse.Namespace) -> list[dict[str, Any]]:
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT boundary_key,boundary_text,enforcement_mode FROM lucidota_control.demem_boundary WHERE active ORDER BY boundary_key")
            return [dict(r) for r in cur.fetchall()]


def decide(instruction: str, boundaries: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], str]:
    hits: list[dict[str, Any]] = []
    for b in boundaries:
        key = b["boundary_key"]
        rx = PATTERNS.get(key)
        if rx and rx.search(instruction):
            hits.append(b)
    if any(h["enforcement_mode"] == "block" for h in hits):
        decision = "block"
    elif any(h["enforcement_mode"] == "rewrite" for h in hits):
        decision = "rewrite"
    elif hits:
        decision = "warn"
    else:
        decision = "allow"
    guarded = instruction
    if hits:
        guarded += "\n\nDeMem boundaries preserved: " + "; ".join(h["boundary_text"] for h in hits)
    return decision, hits, guarded


def check(args: argparse.Namespace) -> int:
    boundaries = active_boundaries(args)
    decision, hits, guarded = decide(args.instruction, boundaries)
    report = {
        "action": "check",
        "execute_performed": bool(args.execute),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "source_ref": args.source_ref,
        "instruction_sha256": sha(args.instruction),
        "decision": decision,
        "boundary_hits": hits,
        "guarded_instruction": guarded,
        "canonical_mutation_allowed": False,
        "blockers": [h["boundary_key"] for h in hits if h["enforcement_mode"] == "block"],
    }
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.demem_enforcement_decision(
                      instruction_sha256, source_ref, boundary_hits, decision, guarded_instruction, detail
                    )
                    VALUES (%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                    RETURNING decision_uuid::text
                    """,
                    (report["instruction_sha256"], args.source_ref, json.dumps(hits), decision, guarded, json.dumps({"script": "scripts/demem_runtime_guard.py"})),
                )
                report["decision_uuid"] = cur.fetchone()[0]
            conn.commit()
        report["db_writes_performed"] = True
    write_report("check_execute" if args.execute else "check_dry_run", report)
    if "decision_uuid" in report:
        print(f"DECISION_UUID={report['decision_uuid']}")
    print(f"DEMEM_DECISION={decision}")
    return 2 if decision == "block" else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Apply DeMem boundary enforcement to an instruction")
    p.add_argument("--database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("check"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--instruction", required=True); sp.add_argument("--source-ref", default="operator_cli"); sp.set_defaults(func=check)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
