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


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-indy-brief")
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--json", action="store_true")
    r = sub.add_parser("remember")
    r.add_argument("title")
    r.add_argument("--kind", choices=["task", "decision", "correction", "note"], default="note")
    r.add_argument("--body", default="")
    r.add_argument("--source", default="operator")
    q = sub.add_parser("queue")
    q.add_argument("title")
    q.add_argument("--item-type", choices=["calendar_intent", "reminder", "wiki_note", "auth_inventory", "review"], default="review")
    q.add_argument("--body", default="")
    q.add_argument("--urgency", choices=["low", "normal", "high"], default="normal")
    q.add_argument("--due-at", default=None)
    args = ap.parse_args()
    if args.cmd == "remember":
        report = remember(args)
    elif args.cmd == "queue":
        report = queue_item(args)
    else:
        report = make_brief()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    elif "now" in report:
        print(render(report))
    elif "memory_id" in report:
        print(f"remembered {report['memory_id']}: {report['title']}")
    elif "queue_id" in report:
        print(f"queued {report['queue_id']}: {report['title']}")
    else:
        print(report)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
