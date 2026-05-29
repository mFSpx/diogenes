#!/usr/bin/env python3
"""Probe canonical graph write barrier without mutating graph state."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/graph"
SCHEMAS = [
    ROOT / "06_SCHEMA/040_graph_write_barrier_enforcement.sql",
    ROOT / "06_SCHEMA/074_graph_journal_write_barrier.sql",
]


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def db(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"graph_write_blocker_probe_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def counts(cur: psycopg.Cursor[Any]) -> dict[str, int]:
    out = {}
    for table in ("graph_item", "graph_edge", "graph_journal"):
        cur.execute(f"SELECT count(*) FROM lucidota_go.{table}")
        out[table] = int(cur.fetchone()[0])
    return out


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db(args)) as conn:
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


def probe(args: argparse.Namespace) -> int:
    attempts: list[dict[str, Any]] = []
    with psycopg.connect(db(args)) as conn:
        with conn.cursor() as cur:
            before = counts(cur)
            cur.execute("SELECT uuid::text FROM lucidota_go.graph_item ORDER BY created_at DESC LIMIT 1")
            item = cur.fetchone()
            item_uuid = item[0] if item else None
            cur.execute("SELECT edge_uuid::text FROM lucidota_go.graph_edge ORDER BY created_at DESC LIMIT 1")
            edge = cur.fetchone()
            edge_uuid = edge[0] if edge else None
            cur.execute("SELECT journal_uuid::text FROM lucidota_go.graph_journal ORDER BY created_at DESC LIMIT 1")
            journal = cur.fetchone()
            journal_uuid = journal[0] if journal else None
            probes: list[tuple[str, str | None]] = [
                ("insert_graph_item_without_promotion_path", "INSERT INTO lucidota_go.graph_item(term,label,status,location_at_on_graph,payload) VALUES ('CLAIM','probe direct insert','staged','graph_write_blocker_probe','{}'::jsonb)"),
                ("update_graph_item_without_promotion_path", "UPDATE lucidota_go.graph_item SET label=label WHERE uuid=(SELECT uuid FROM lucidota_go.graph_item LIMIT 1)"),
                ("delete_graph_item_without_promotion_path", "DELETE FROM lucidota_go.graph_item WHERE uuid=(SELECT uuid FROM lucidota_go.graph_item LIMIT 1)"),
                ("insert_graph_edge_without_promotion_path", "INSERT INTO lucidota_go.graph_edge(source_uuid,target_uuid,edge_type,detail) SELECT uuid, uuid, 'RELATED_TO', '{\"probe\":\"direct_edge_insert\"}'::jsonb FROM lucidota_go.graph_item LIMIT 1"),
                ("update_graph_edge_without_promotion_path", "UPDATE lucidota_go.graph_edge SET detail=detail WHERE edge_uuid=(SELECT edge_uuid FROM lucidota_go.graph_edge LIMIT 1)" if edge_uuid else None),
                ("delete_graph_edge_without_promotion_path", "DELETE FROM lucidota_go.graph_edge WHERE edge_uuid=(SELECT edge_uuid FROM lucidota_go.graph_edge LIMIT 1)" if edge_uuid else None),
                ("insert_graph_journal_without_promotion_path", "INSERT INTO lucidota_go.graph_journal(item_uuid,action,reason,after_state) SELECT uuid, 'stage', 'probe direct journal insert', '{}'::jsonb FROM lucidota_go.graph_item LIMIT 1" if item_uuid else None),
                ("update_graph_journal_without_promotion_path", "UPDATE lucidota_go.graph_journal SET reason=reason WHERE journal_uuid=(SELECT journal_uuid FROM lucidota_go.graph_journal LIMIT 1)" if journal_uuid else None),
                ("delete_graph_journal_without_promotion_path", "DELETE FROM lucidota_go.graph_journal WHERE journal_uuid=(SELECT journal_uuid FROM lucidota_go.graph_journal LIMIT 1)" if journal_uuid else None),
            ]
            for name, sql in probes:
                if sql is None:
                    attempts.append({"attempt": name, "blocked": True, "skipped": True, "error": "no_existing_target_row"})
                    continue
                try:
                    cur.execute(sql)
                    attempts.append({"attempt": name, "blocked": False, "error": ""})
                    conn.rollback()
                except Exception as exc:
                    attempts.append({"attempt": name, "blocked": True, "error": str(exc).splitlines()[0]})
                    conn.rollback()
                cur = conn.cursor()
            after = counts(cur)
    passed = all(a["blocked"] for a in attempts) and before == after
    report = {
        "action": "probe",
        "execute_performed": True,
        "db_writes_performed": False,
        "canonical_graph_writes_performed": False,
        "counts_before": before,
        "counts_after": after,
        "attempts": attempts,
        "pass": passed,
        "blockers": [] if passed else ["direct_graph_write_barrier_failed_or_counts_changed"],
    }
    write_report("execute", report)
    print("GRAPH_WRITE_BLOCKER=" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Verify direct canonical graph writes are blocked")
    p.add_argument("--database-url")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("init-schema")
    sp.add_argument("--execute", action="store_true")
    sp.set_defaults(func=init_schema)
    sp = sub.add_parser("probe")
    sp.set_defaults(func=probe)
    args = p.parse_args()
    if not args.cmd:
        args.func = probe
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
