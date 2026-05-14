#!/usr/bin/env python3
"""Indy_Reads runtime brief + quiet task memory.

This is the first product-facing persona adapter: it reads only local project-brain
files and local Postgres tables, then emits a concise, cited build brief. It does
not browse Drive, Gmail, Calendar, or external sources.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import psycopg

from lucidota_indy_corpus import build_corpus

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "00_PROJECT_BRAIN" / "STATUS.md"
TODO = ROOT / "00_PROJECT_BRAIN" / "TODO.md"
AUDIT = ROOT / "00_PROJECT_BRAIN" / "BUILD_PLAN_AUDIT.md"
CONTRACT = ROOT / "00_PROJECT_BRAIN" / "INDY_READS_RUNTIME_CONTRACT.md"
SCHEMA = ROOT / "06_SCHEMA" / "014_indy_runtime.sql"
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")


def apply_schema(conn) -> None:
    conn.execute(SCHEMA.read_text(encoding="utf-8"))


def section(path: Path, heading: str) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    target = heading.strip().lower()
    for line in lines:
        if line.startswith("## "):
            name = line.lstrip("#").strip().lower()
            if in_section and name != target:
                break
            in_section = name == target
            continue
        if in_section:
            out.append(line)
    return out


def bullets(lines: list[str], limit: int) -> list[str]:
    found = []
    for ln in lines:
        x = ln.strip()
        if x.startswith("- "):
            item = x[2:].strip()
            item = re.sub(r"^\[[ xX]\]\s*", "", item)
            found.append(item)
    return [re.sub(r"\s+", " ", x) for x in found[:limit]]


def audit_phase(name: str) -> dict:
    text = AUDIT.read_text(encoding="utf-8", errors="ignore") if AUDIT.exists() else ""
    m = re.search(rf"^###\s+[^\n]*{re.escape(name)}[^\n]*—\s*([^\n]+)\n(?P<body>.*?)(?=\n### |\Z)", text, re.S | re.I | re.M)
    if not m:
        return {"bar": "unknown", "done": [], "open": []}
    body = m.group("body")
    done = re.findall(r"^- \[x\] (.+)$", body, re.M)
    open_ = re.findall(r"^- \[ \] (.+)$", body, re.M)
    return {"bar": m.group(1).strip(), "done": done[:6], "open": open_[:8]}


def db_counts() -> dict:
    queries = {
        "active_memory": "SELECT count(*) FROM lucidota_indy.task_memory WHERE status='active'",
        "quiet_queue": "SELECT count(*) FROM lucidota_indy.side_queue WHERE status='queued'",
        "workflow_events": "SELECT count(*) FROM lucidota_control.workflow_event",
        "auth_records": "SELECT count(*) FROM lucidota_indy.auth_inventory",
    }
    try:
        with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
            apply_schema(conn)
            conn.commit()
            return {k: conn.execute(sql).fetchone()[0] for k, sql in queries.items()}
    except Exception as exc:
        return {"db_error": str(exc)[:160]}


def recent_memory(limit: int = 5) -> list[dict]:
    try:
        with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
            apply_schema(conn)
            rows = conn.execute(
                """
                SELECT kind, title, status, created_at::text
                FROM lucidota_indy.task_memory
                WHERE status='active'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            conn.commit()
        return [dict(zip(["kind", "title", "status", "created_at"], r)) for r in rows]
    except Exception:
        return []



def recent_queue(limit: int = 5) -> list[dict]:
    try:
        with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
            apply_schema(conn)
            rows = conn.execute(
                """
                SELECT item_type, title, urgency, status, COALESCE(due_at::text, '') AS due_at, created_at::text
                FROM lucidota_indy.side_queue
                WHERE status IN ('queued','reviewing')
                ORDER BY CASE urgency WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END, created_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            conn.commit()
        return [dict(zip(["item_type", "title", "urgency", "status", "due_at", "created_at"], r)) for r in rows]
    except Exception:
        return []


def auth_records(limit: int = 8) -> list[dict]:
    try:
        with psycopg.connect(STATE_DB, connect_timeout=3) as conn:
            apply_schema(conn)
            rows = conn.execute(
                """
                SELECT service, account_hint, access_status, scope_note, updated_at::text
                FROM lucidota_indy.auth_inventory
                ORDER BY service, account_hint
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            conn.commit()
        return [dict(zip(["service", "account_hint", "access_status", "scope_note", "updated_at"], r)) for r in rows]
    except Exception:
        return []

def corpus_summary() -> dict:
    corpus = build_corpus()
    return {
        "artifact": corpus["artifact"],
        "unit_count": corpus["unit_count"],
        "artifact_sha256": corpus["artifact_sha256"],
        "source_scope": corpus["source_scope"],
        "labels": corpus["labels"],
    }


def make_brief() -> dict:
    status_next = bullets(section(STATUS, "Next Verification"), 10)
    todo_next = bullets(section(TODO, "Next"), 10)
    indy = audit_phase("Indy_Reads")
    return {
        "ok": True,
        "identity": "Indy_Reads",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "now": status_next[:6],
        "next": todo_next[:6],
        "indy_phase": indy,
        "memory": recent_memory(),
        "queue": recent_queue(),
        "auth": auth_records(),
        "corpus": corpus_summary(),
        "counters": db_counts(),
        "citations": [
            "00_PROJECT_BRAIN/STATUS.md#next-verification",
            "00_PROJECT_BRAIN/TODO.md#next",
            "00_PROJECT_BRAIN/BUILD_PLAN_AUDIT.md#011-indy_reads--persona--assistant-layer",
            "00_PROJECT_BRAIN/INDY_READS_RUNTIME_CONTRACT.md",
        ],
    }


def render(report: dict) -> str:
    lines = ["INDY_READS BRIEF", "================", "Mode: local project brain + Postgres only", ""]
    lines += ["Now / verified edge:"] + [f"  - {x}" for x in report["now"]]
    lines += ["", "Next moves:"] + [f"  - {x}" for x in report["next"]]
    phase = report["indy_phase"]
    lines += ["", f"Persona phase: {phase['bar']}"]
    if phase.get("open"):
        lines += ["Open persona items:"] + [f"  - {x}" for x in phase["open"][:5]]
    lines += ["", "Quiet memory:"]
    if report["memory"]:
        lines += [f"  - [{m['kind']}] {m['title']}" for m in report["memory"]]
    else:
        lines += ["  - none yet"]
    lines += ["", "Quiet queue:"]
    if report.get("queue"):
        lines += [f"  - [{q['urgency']}/{q['item_type']}] {q['title']}" for q in report["queue"]]
    else:
        lines += ["  - none queued"]
    corpus = report.get("corpus", {})
    lines += ["", "Persona corpus:"]
    if corpus:
        lines += [f"  - {corpus.get('artifact')}: {corpus.get('unit_count')} units ({corpus.get('artifact_sha256')})"]
    else:
        lines += ["  - unavailable"]
    lines += ["", "Auth inventory:"]
    if report.get("auth"):
        lines += [f"  - {a['service']}: {a['access_status']} ({a['account_hint'] or 'no account hint'})" for a in report["auth"]]
    else:
        lines += ["  - none recorded"]
    lines += ["", "Counters:"] + [f"  - {k}: {v}" for k, v in report["counters"].items()]
    lines += ["", "Citations:"] + [f"  - {c}" for c in report["citations"]]
    return "\n".join(lines)


def remember(args) -> dict:
    with psycopg.connect(STATE_DB) as conn:
        apply_schema(conn)
        row = conn.execute(
            """
            INSERT INTO lucidota_indy.task_memory (kind, title, body, source, evidence)
            VALUES (%s,%s,%s,%s,%s::jsonb)
            RETURNING memory_id::text
            """,
            (args.kind, args.title, args.body or "", args.source, json.dumps({"cli": "lucidota_indy_brief.py remember"})),
        ).fetchone()
        conn.commit()
    return {"ok": True, "memory_id": row[0], "title": args.title}


def queue_item(args) -> dict:
    with psycopg.connect(STATE_DB) as conn:
        apply_schema(conn)
        row = conn.execute(
            """
            INSERT INTO lucidota_indy.side_queue (item_type, title, body, urgency, due_at, evidence)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb)
            RETURNING queue_id::text
            """,
            (args.item_type, args.title, args.body or "", args.urgency, args.due_at, json.dumps({"quiet_side_queue": True})),
        ).fetchone()
        conn.commit()
    return {"ok": True, "queue_id": row[0], "title": args.title}




def reminder(args) -> dict:
    args.item_type = "reminder"
    return queue_item(args)


def calendar_intent(args) -> dict:
    args.item_type = "calendar_intent"
    return queue_item(args)

def correction(args) -> dict:
    args.kind = "correction"
    args.source = args.source or "operator"
    return remember(args)


def list_queue(args) -> dict:
    items = recent_queue(args.limit)
    return {"ok": True, "queue": items}


def mark_queue(args) -> dict:
    with psycopg.connect(STATE_DB) as conn:
        apply_schema(conn)
        row = conn.execute(
            """
            UPDATE lucidota_indy.side_queue
            SET status=%s, updated_at=now()
            WHERE queue_id=%s::uuid
            RETURNING queue_id::text, title, status
            """,
            (args.status, args.queue_id),
        ).fetchone()
        conn.commit()
    return {"ok": row is not None, "queue_id": row[0] if row else args.queue_id, "title": row[1] if row else "", "status": row[2] if row else "missing"}


def auth_add(args) -> dict:
    # Never store raw secrets here. secret_ref is a pointer/label only.
    with psycopg.connect(STATE_DB) as conn:
        apply_schema(conn)
        row = conn.execute(
            """
            INSERT INTO lucidota_indy.auth_inventory
              (service, account_hint, access_status, scope_note, secret_ref, evidence)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (service, account_hint) DO UPDATE SET
              access_status=EXCLUDED.access_status,
              scope_note=EXCLUDED.scope_note,
              secret_ref=EXCLUDED.secret_ref,
              evidence=EXCLUDED.evidence,
              updated_at=now()
            RETURNING auth_id::text
            """,
            (args.service, args.account_hint or "", args.status, args.scope_note or "", args.secret_ref or "", json.dumps({"no_raw_secret": True, "source": "indy_auth_add"})),
        ).fetchone()
        conn.commit()
    return {"ok": True, "auth_id": row[0], "service": args.service, "status": args.status}


def auth_list(args) -> dict:
    return {"ok": True, "auth": auth_records(args.limit)}

def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-indy-brief")
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--json", action="store_true")
    r = sub.add_parser("remember")
    r.add_argument("--json", action="store_true")
    r.add_argument("title")
    r.add_argument("--kind", choices=["task", "decision", "correction", "note"], default="note")
    r.add_argument("--body", default="")
    r.add_argument("--source", default="operator")
    c = sub.add_parser("correct")
    c.add_argument("--json", action="store_true")
    c.add_argument("title")
    c.add_argument("--body", default="")
    c.add_argument("--source", default="operator")
    q = sub.add_parser("queue")
    q.add_argument("--json", action="store_true")
    q.add_argument("title")
    q.add_argument("--item-type", choices=["calendar_intent", "reminder", "wiki_note", "auth_inventory", "review"], default="review")
    q.add_argument("--body", default="")
    q.add_argument("--urgency", choices=["low", "normal", "high"], default="normal")
    q.add_argument("--due-at", default=None)
    rem = sub.add_parser("reminder")
    rem.add_argument("--json", action="store_true")
    rem.add_argument("title")
    rem.add_argument("--body", default="")
    rem.add_argument("--urgency", choices=["low", "normal", "high"], default="normal")
    rem.add_argument("--due-at", default=None)
    cal = sub.add_parser("calendar-intent")
    cal.add_argument("--json", action="store_true")
    cal.add_argument("title")
    cal.add_argument("--body", default="")
    cal.add_argument("--urgency", choices=["low", "normal", "high"], default="normal")
    cal.add_argument("--due-at", default=None)
    lq = sub.add_parser("queue-list")
    lq.add_argument("--json", action="store_true")
    lq.add_argument("--limit", type=int, default=10)
    mq = sub.add_parser("queue-mark")
    mq.add_argument("--json", action="store_true")
    mq.add_argument("queue_id")
    mq.add_argument("--status", choices=["queued", "reviewing", "done", "dismissed"], default="done")
    aa = sub.add_parser("auth-add")
    aa.add_argument("--json", action="store_true")
    aa.add_argument("service")
    aa.add_argument("--account-hint", default="")
    aa.add_argument("--status", choices=["unknown", "missing", "available", "blocked", "not_exposed"], default="unknown")
    aa.add_argument("--scope-note", default="")
    aa.add_argument("--secret-ref", default="")
    al = sub.add_parser("auth-list")
    al.add_argument("--json", action="store_true")
    al.add_argument("--limit", type=int, default=20)
    co = sub.add_parser("corpus")
    co.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.cmd == "remember":
        report = remember(args)
    elif args.cmd == "correct":
        report = correction(args)
    elif args.cmd == "queue":
        report = queue_item(args)
    elif args.cmd == "reminder":
        report = reminder(args)
    elif args.cmd == "calendar-intent":
        report = calendar_intent(args)
    elif args.cmd == "queue-list":
        report = list_queue(args)
    elif args.cmd == "queue-mark":
        report = mark_queue(args)
    elif args.cmd == "auth-add":
        report = auth_add(args)
    elif args.cmd == "auth-list":
        report = auth_list(args)
    elif args.cmd == "corpus":
        report = build_corpus()
    else:
        report = make_brief()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    elif "now" in report:
        print(render(report))
    elif "memory_id" in report:
        print(f"remembered {report['memory_id']}: {report['title']}")
    elif "queue_id" in report:
        action = "queued" if "status" not in report else report["status"]
        print(f"{action} {report['queue_id']}: {report['title']}")
    elif "auth_id" in report:
        print(f"auth {report['service']}: {report['status']} ({report['auth_id']})")
    elif "artifact" in report and "unit_count" in report:
        print(f"{report['artifact']}: {report['unit_count']} units {report.get('artifact_sha256', '')}")
    elif "queue" in report:
        print("\n".join(f"{q['status']} {q['urgency']} {q['item_type']} {q['title']} {q['due_at']}" for q in report["queue"]))
    elif "auth" in report:
        print("\n".join(f"{a['service']} {a['access_status']} {a['account_hint']} {a['scope_note']}" for a in report["auth"]))
    else:
        print(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
