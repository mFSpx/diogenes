#!/usr/bin/env python3
"""Decision-quality delta analyzer for KORPUS/chat timelines.

This does not diagnose a person. It measures text evidence of decision hygiene:
evidence before action, planning, delay/de-escalation, support seeking,
boundaries, completion/outcome language, and impulsive/risk/scarcity markers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
SCHEMA = ROOT / "06_SCHEMA" / "020_chat_dump_timeline.sql"

import sys
sys.path.insert(0, str(ROOT))
from ALGOS.decision_hygiene import counts, score_features, excerpt, monthly, compare_halves  # type: ignore  # noqa: E402

def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def scan_chat(conn: psycopg.Connection, limit: int, persist: bool) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT message_uuid::text, provider, create_time, content_text
        FROM lucidota_chatdump.message
        WHERE role IN ('user','human','') AND content_text <> ''
        ORDER BY create_time ASC NULLS LAST, created_at ASC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    out = []
    for uuid, provider, occurred_at, text in rows:
        c = counts(text)
        score, label = score_features(c)
        rec = {"source_kind": "chat_message", "source_uuid": uuid, "provider": provider, "occurred_at": occurred_at, "score": score, "label": label, **c, "excerpt": excerpt(text)}
        out.append(rec)
        if persist:
            persist_signal(conn, rec)
    return out


def scan_korpus(conn: psycopg.Connection, limit: int, persist: bool) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT c.component_uuid::text, fo.first_seen_at, c.content
        FROM lucidota_korpus.component c
        JOIN lucidota_korpus.file_object fo ON fo.file_uuid=c.file_uuid
        WHERE c.content <> ''
        ORDER BY fo.first_seen_at ASC, c.created_at ASC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    out = []
    for uuid, occurred_at, text in rows:
        c = counts(text)
        score, label = score_features(c)
        rec = {"source_kind": "korpus_component", "source_uuid": uuid, "provider": "korpus", "occurred_at": occurred_at, "score": score, "label": label, **c, "excerpt": excerpt(text)}
        out.append(rec)
        if persist:
            persist_signal(conn, rec)
    return out


def scan_commdump(conn: psycopg.Connection, limit: int, persist: bool) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT message_uuid::text, source_kind, occurred_at, COALESCE(subject || ' ', '') || content_text AS text
        FROM lucidota_commdump.message
        WHERE content_text <> ''
        ORDER BY occurred_at ASC NULLS LAST, created_at ASC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    out = []
    for uuid, provider, occurred_at, text in rows:
        c = counts(text)
        score, label = score_features(c)
        rec = {"source_kind": "comm_message", "source_uuid": uuid, "provider": provider, "occurred_at": occurred_at, "score": score, "label": label, **c, "excerpt": excerpt(text)}
        out.append(rec)
        if persist:
            persist_signal(conn, rec)
    return out


def persist_signal(conn: psycopg.Connection, rec: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO lucidota_chatdump.decision_signal(
          source_kind, source_uuid, provider, occurred_at, score,
          evidence_count, planning_count, delay_count, support_count, boundary_count, outcome_count,
          impulsive_count, scarcity_count, risk_count, label, features, excerpt
        ) VALUES (%s,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
        ON CONFLICT(source_kind, source_uuid) DO UPDATE SET
          provider=EXCLUDED.provider, occurred_at=EXCLUDED.occurred_at, score=EXCLUDED.score,
          evidence_count=EXCLUDED.evidence_count, planning_count=EXCLUDED.planning_count,
          delay_count=EXCLUDED.delay_count, support_count=EXCLUDED.support_count,
          boundary_count=EXCLUDED.boundary_count, outcome_count=EXCLUDED.outcome_count,
          impulsive_count=EXCLUDED.impulsive_count, scarcity_count=EXCLUDED.scarcity_count,
          risk_count=EXCLUDED.risk_count, label=EXCLUDED.label, features=EXCLUDED.features, excerpt=EXCLUDED.excerpt
        """,
        (
            rec["source_kind"], rec["source_uuid"], rec["provider"], rec["occurred_at"], rec["score"],
            rec["evidence_count"], rec["planning_count"], rec["delay_count"], rec["support_count"], rec["boundary_count"], rec["outcome_count"],
            rec["impulsive_count"], rec["scarcity_count"], rec["risk_count"], rec["label"],
            jdump({k: rec[k] for k in rec if k.endswith("_count")}), rec["excerpt"],
        ),
    )


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        rows: list[dict[str, Any]] = []
        if args.source in {"chat", "both"}:
            rows.extend(scan_chat(conn, args.limit, not args.no_persist))
        if args.source in {"comm", "both"}:
            rows.extend(scan_commdump(conn, args.limit, not args.no_persist))
        if args.source in {"korpus", "both"}:
            rows.extend(scan_korpus(conn, args.limit, not args.no_persist))
        if not args.no_persist:
            conn.commit()
    return {"ok": True, "signals": len(rows), "comparison": compare_halves(rows), "monthly": monthly(rows), "examples": sorted(rows, key=lambda r: r["score"], reverse=True)[: args.examples]}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_schema(conn)
        row = conn.execute("SELECT count(*), min(occurred_at)::text, max(occurred_at)::text, avg(score)::numeric(10,2) FROM lucidota_chatdump.decision_signal").fetchone()
        by_label = conn.execute("SELECT label, count(*), avg(score)::numeric(10,2) FROM lucidota_chatdump.decision_signal GROUP BY label ORDER BY count(*) DESC, label").fetchall()
    return {"ok": True, "status": {"signals": row[0], "earliest": row[1], "latest": row[2], "avg_score": row[3]}, "labels": [{"label": r[0], "n": r[1], "avg_score": r[2]} for r in by_label]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lucidota-decision-delta")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--source", choices=["chat", "comm", "korpus", "both"], default="both")
    p.add_argument("--limit", type=int, default=1000000)
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--no-persist", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
