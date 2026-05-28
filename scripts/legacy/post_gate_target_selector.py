#!/usr/bin/env python3
"""Post Mega-Gate target selector.

Executable target-selection framework:
- Reads TICKLETRUNK and status ledger before decisions.
- Scores/selects targets deterministically.
- Requires evidence refs for selected/executed states.
- Writes DB audit rows and work-loop receipts only under --execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA/104_post_gate_target_selection.sql"
OUT = ROOT / "05_OUTPUTS/post_gate"
LEDGER = ROOT / "05_OUTPUTS/status_ledger.json"
TICK = ROOT / "00_PROJECT_BRAIN/TICKLETRUNK.json"
WORK_LOOP_LEDGER = ROOT / "05_OUTPUTS/work_loops/real_code_loop_ledger.jsonl"
VALID_STATES = {"candidate", "selected", "deferred", "blocked", "executed", "archived"}
ALLOWED_TRANSITIONS = {
    ("candidate", "selected"), ("candidate", "deferred"), ("candidate", "blocked"), ("candidate", "archived"),
    ("selected", "executed"), ("selected", "deferred"), ("selected", "blocked"), ("selected", "archived"),
    ("deferred", "selected"), ("deferred", "blocked"), ("deferred", "archived"),
    ("blocked", "deferred"), ("blocked", "selected"), ("blocked", "archived"),
    ("executed", "archived"), ("archived", "archived"),
}
RISK_WEIGHT = {"low": 0, "medium": 20, "high": 45, "destructive": 1000}
STATE_WEIGHT = {"candidate": 0, "selected": -30, "deferred": 25, "blocked": 100, "executed": 500, "archived": 900}
REQUIRED_REPORT_FIELDS = {"action", "execute_performed", "blockers", "status", "report_path"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(p: str | Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def evidence_ok(ref: str) -> bool:
    if ":" in ref and not ref.startswith((".", "/")):
        return True
    p = ROOT / ref if not Path(ref).is_absolute() else Path(ref)
    return p.exists()


def normalize_blocker(raw: str) -> str:
    text = raw.strip().upper().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in "_:.") or "UNKNOWN_BLOCKER"


def validate_worker_mapping(mapping: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    scripts = mapping.get("scripts", []) if isinstance(mapping, dict) else []
    if not scripts:
        blockers.append("WORKER_MAPPING_SCRIPTS_MISSING")
    for script in scripts:
        p = ROOT / str(script)
        if not p.exists():
            blockers.append(f"WORKER_SCRIPT_NOT_FOUND:{script}")
    return blockers


def ensure_context() -> dict[str, Any]:
    tick = read_json(TICK)
    ledger = read_json(LEDGER)
    next_actions = [e.get("next_action") for sec in ("software", "phases", "plans_workstreams") for e in ledger.get(sec, []) if e.get("next_action")]
    hard_laws = [law.get("law_key") or law.get("title") for law in ledger.get("hard_laws", [])]
    return {"tickletrunk_total_tools": tick.get("total_tools"), "ledger_next_actions": next_actions, "hard_laws": hard_laws}


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"post_gate_target_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    payload.setdefault("schema", "lucidota.post_gate_target.report.v2")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def append_work_loop_receipt(payload: dict[str, Any], report_path: Path) -> None:
    WORK_LOOP_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "loop": "post_gate",
        "item": payload.get("action"),
        "target": payload.get("target_key") or payload.get("recommended") or "post_gate_target_selection",
        "counted": True,
        "files_changed": ["scripts/post_gate_target_selector.py", "06_SCHEMA/104_post_gate_target_selection.sql"],
        "validation": [{"command": "post_gate_target_selector", "result": payload.get("status"), "evidence": rel(report_path)}],
        "capability_delta": "Post-gate target selection state/audit/report path updated from executable selector.",
        "blocked_by": ";".join(payload.get("blockers") or []) or None,
        "next_action": payload.get("next_action") or "continue selected target",
        "created_at": now(),
    }
    WORK_LOOP_LEDGER.open("a", encoding="utf-8").write(json.dumps(entry, default=str) + "\n")


def apply_schema(cur: Any) -> None:
    cur.execute(SCHEMA.read_text(encoding="utf-8"))


def score_target(row: dict[str, Any]) -> int:
    evidence_refs = row.get("evidence_refs") or []
    worker_mapping = row.get("worker_mapping") or {}
    missing_worker_penalty = 100 if validate_worker_mapping(worker_mapping) else 0
    evidence_penalty = 75 if not evidence_refs else 0
    gate_penalty = 15 if row.get("requires_operator_gate") else 0
    return int(row.get("priority") or 100) + RISK_WEIGHT.get(row.get("risk_class"), 100) + STATE_WEIGHT.get(row.get("target_state"), 100) + missing_worker_penalty + evidence_penalty + gate_penalty


def validate_report_shape(path: Path) -> list[str]:
    blockers: list[str] = []
    try:
        data = read_json(path)
    except Exception as exc:
        return [f"REPORT_JSON_UNREADABLE:{exc}"]
    missing = sorted(REQUIRED_REPORT_FIELDS - set(data.keys()))
    if missing:
        blockers.append("REPORT_FIELDS_MISSING:" + ",".join(missing))
    if data.get("status") not in {"PASS", "FAIL"}:
        blockers.append("REPORT_STATUS_INVALID")
    if not isinstance(data.get("blockers", []), list):
        blockers.append("REPORT_BLOCKERS_NOT_LIST")
    rp = data.get("report_path")
    if rp and not (ROOT / rp).exists():
        blockers.append("REPORT_PATH_NOT_READABLE")
    return blockers


def init(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
            with conn.cursor() as cur:
                apply_schema(cur)
            conn.commit()
    payload = {"action": "init_schema", "execute_performed": bool(args.execute), "schema_path": rel(SCHEMA), **ensure_context(), "blockers": [], "status": "PASS"}
    path = write_report("init_schema_execute" if args.execute else "init_schema_dry_run", payload)
    if args.execute:
        append_work_loop_receipt(payload, path)
    return 0


def list_targets(args: argparse.Namespace) -> int:
    ctx = ensure_context()
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            apply_schema(cur)
            cur.execute("""SELECT target_uuid::text,target_key,subsystem,title,priority,risk_class,requires_operator_gate,target_state,evidence_refs,worker_mapping,blocker
                           FROM lucidota_control.post_gate_target ORDER BY priority,target_key""")
            rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
    for r in rows:
        r["worker_mapping_blockers"] = validate_worker_mapping(r.get("worker_mapping") or {})
        r["evidence_missing"] = [ref for ref in (r.get("evidence_refs") or []) if not evidence_ok(str(ref))]
        r["score"] = score_target(r)
    recommended = min(rows, key=lambda r: (r["score"], r["target_key"]))["target_key"] if rows else None
    blockers = [] if rows else ["NO_TARGETS_REGISTERED"]
    payload = {"action": "list", "execute_performed": False, "targets": rows, "recommended": recommended, **ctx, "blockers": blockers, "status": "PASS" if not blockers else "FAIL"}
    write_report("list", payload)
    print("TARGETS=" + str(len(rows)))
    return 0 if not blockers else 4


def transition(args: argparse.Namespace, state: str) -> int:
    ctx = ensure_context()
    refs = [x.strip() for x in (args.evidence_ref or []) if x.strip()]
    blockers: list[str] = []
    if state in {"selected", "executed"} and not refs:
        blockers.append("MISSING_EVIDENCE_REF")
    for r in refs:
        if not evidence_ok(r):
            blockers.append("EVIDENCE_REF_NOT_FOUND:" + r)
    target: dict[str, Any] = {}
    audit_count = 0
    idempotent_noop = False
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            apply_schema(cur)
            cur.execute("SELECT * FROM lucidota_control.post_gate_target WHERE target_key=%s FOR UPDATE", (args.target_key,))
            row = cur.fetchone()
            target = dict(row or {})
            if not target:
                blockers.append("TARGET_NOT_FOUND")
            else:
                old = target["target_state"]
                if old == state:
                    idempotent_noop = True
                elif (old, state) not in ALLOWED_TRANSITIONS:
                    blockers.append(f"ILLEGAL_TRANSITION:{old}->{state}")
                if target.get("risk_class") == "destructive" and state == "selected":
                    blockers.append("OPERATOR_GATE_REQUIRED")
                blockers.extend(validate_worker_mapping(target.get("worker_mapping") or {}))
                if args.blocker:
                    args.blocker = normalize_blocker(args.blocker)
                if args.execute and not blockers and not idempotent_noop:
                    cur.execute("""UPDATE lucidota_control.post_gate_target
                                   SET target_state=%s,evidence_refs=%s::jsonb,blocker=%s
                                   WHERE target_key=%s RETURNING *""", (state, json.dumps(refs or target.get("evidence_refs") or []), args.blocker or "", args.target_key))
                    target = dict(cur.fetchone())
                    cur.execute("""INSERT INTO lucidota_control.post_gate_target_audit(target_uuid,target_key,old_state,new_state,evidence_refs,detail)
                                   VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb)""", (target["target_uuid"], args.target_key, old, state, json.dumps(refs), json.dumps({"script": "scripts/post_gate_target_selector.py", "idempotent_noop": False})))
                    audit_count = 1
        conn.commit()
    blockers = [normalize_blocker(b) if ":" not in b else b for b in blockers]
    payload = {"action": state, "execute_performed": bool(args.execute), "target_key": args.target_key, "target": target, "audit_rows_written": audit_count, "idempotent_noop": idempotent_noop, "evidence_refs": refs, **ctx, "blockers": blockers, "status": "PASS" if not blockers else "FAIL", "next_action": f"execute_target:{args.target_key}" if state == "selected" else "continue_post_gate_selection"}
    path = write_report(state + ("_execute" if args.execute else "_dry_run"), payload)
    if args.execute:
        append_work_loop_receipt(payload, path)
    print("POST_GATE_TARGET=" + payload["status"])
    return 0 if not blockers else 4


def selftest(args: argparse.Namespace) -> int:
    key = "self_test_" + hashlib.sha256(stamp().encode()).hexdigest()[:8]
    blockers: list[str] = []
    uuid = None
    audits = 0
    with psycopg.connect(db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            apply_schema(cur)
            if args.execute:
                cur.execute("""INSERT INTO lucidota_control.post_gate_target(target_key,subsystem,title,priority,risk_class,evidence_refs,worker_mapping)
                               VALUES (%s,'selftest','selector self test',1,'low','[\"scripts/post_gate_target_selector.py\"]','{"scripts":["scripts/post_gate_target_selector.py"]}') RETURNING target_uuid::text""", (key,))
                uuid = cur.fetchone()["target_uuid"]
                for st in ("selected", "executed", "archived"):
                    cur.execute("UPDATE lucidota_control.post_gate_target SET target_state=%s WHERE target_key=%s", (st, key))
                cur.execute("SELECT count(*) n FROM lucidota_control.post_gate_target_audit WHERE target_key=%s", (key,))
                audits = int(cur.fetchone()["n"])
                if audits < 3:
                    blockers.append("AUDIT_ROWS_MISSING")
        conn.commit()
    payload = {"action": "self_test", "execute_performed": bool(args.execute), "target_key": key, "target_uuid": uuid, "audit_rows": audits, "blockers": blockers, "status": "PASS" if not blockers else "FAIL"}
    path = write_report("self_test_execute" if args.execute else "self_test_dry_run", payload)
    if args.execute:
        append_work_loop_receipt(payload, path)
    print("POST_GATE_SELF_TEST=" + payload["status"])
    return 0 if not blockers else 4


def validate_report(args: argparse.Namespace) -> int:
    path = ROOT / args.report if not Path(args.report).is_absolute() else Path(args.report)
    blockers = validate_report_shape(path)
    payload = {"action": "validate_report", "execute_performed": False, "validated_report": rel(path), "blockers": blockers, "status": "PASS" if not blockers else "FAIL"}
    write_report("validate_report", payload)
    print("POST_GATE_REPORT_VALID=" + payload["status"])
    return 0 if not blockers else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init)
    sp = sub.add_parser("list"); sp.set_defaults(func=list_targets)
    for name in ("select", "defer", "block", "execute"):
        sp = sub.add_parser(name)
        sp.add_argument("--target-key", required=True)
        sp.add_argument("--execute", action="store_true")
        sp.add_argument("--evidence-ref", action="append", default=[])
        sp.add_argument("--blocker", default="")
        state = "executed" if name == "execute" else "deferred" if name == "defer" else "blocked" if name == "block" else "selected"
        sp.set_defaults(func=lambda a, st=state: transition(a, st))
    sp = sub.add_parser("self-test"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=selftest)
    sp = sub.add_parser("validate-report"); sp.add_argument("--report", required=True); sp.set_defaults(func=validate_report)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
