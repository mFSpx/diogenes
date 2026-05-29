#!/usr/bin/env python3
"""Executable SimpleMem candidate index.

Indexes high-recall candidates from existing claim/design-atom tables into
lucidota_control.simplemem_candidate. Every row is constrained as NOT_TRUTH and
promotion_allowed=false; this is recall substrate, not answer authority.
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
SCHEMA = ROOT / "06_SCHEMA/053_simplemem_candidate_index.sql"
OUT = ROOT / "05_OUTPUTS/simplemem"


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


def state_db(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_db(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"simplemem_candidate_index_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"execute_performed": bool(args.execute), "schema": rel(SCHEMA)})
    return 0


def token_score(query: str, text: str) -> int:
    qt = {t for t in query.lower().split() if len(t) > 2}
    lt = text.lower()
    return sum(1 for t in qt if t in lt)


def fetch_source_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with psycopg.connect(storage_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if args.source in {"claim_packets", "all"}:
                cur.execute(
                    """
                    SELECT 'document_claim_packet:' || claim_packet_uuid::text AS source_ref,
                           claim_text AS candidate_text,
                           confidence_bps::numeric / 10000.0 AS source_score
                    FROM lucidota_korpus.document_claim_packet
                    WHERE truth_status='not_truth_claim_candidate'
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (args.limit,),
                )
                rows.extend(dict(r) for r in cur.fetchall())
            if args.source in {"design_atoms", "all"}:
                cur.execute(
                    """
                    SELECT 'design_atom:' || atom_uuid::text AS source_ref,
                           normalized_claim AS candidate_text,
                           confidence_bps::numeric / 10000.0 AS source_score
                    FROM lucidota_archaeology.design_atom
                    WHERE status='candidate'
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (args.limit,),
                )
                rows.extend(dict(r) for r in cur.fetchall())
    scored = []
    for r in rows:
        score = token_score(args.query, str(r["candidate_text"])) + float(r.get("source_score") or 0)
        if score > 0 or args.include_zero_score:
            r["source_score"] = float(r.get("source_score") or 0)
            r["recall_score"] = float(score)
            scored.append(r)
    return sorted(scored, key=lambda x: x["recall_score"], reverse=True)[: args.limit]


def index(args: argparse.Namespace) -> int:
    query_sha = sha(args.query)
    candidates = fetch_source_candidates(args)
    inserted = 0
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                for cand in candidates:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.simplemem_candidate(query_sha256, source_ref, candidate_text, recall_score, detail)
                        VALUES (%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT (query_sha256, source_ref, md5(candidate_text)) DO NOTHING
                        RETURNING candidate_uuid::text
                        """,
                        (
                            query_sha,
                            cand["source_ref"],
                            cand["candidate_text"],
                            cand["recall_score"],
                            json.dumps({"script": "scripts/simplemem_candidate_index.py", "not_truth": True, "safe_to_answer_from_this_alone": False}),
                        ),
                    )
                    if cur.fetchone() is not None:
                        inserted += 1
                cur.execute(
                    """
                    INSERT INTO lucidota_control.simplemem_candidate_index_run(query_text_sha256, source_table, candidates_seen, candidates_inserted, detail)
                    VALUES (%s,%s,%s,%s,%s::jsonb)
                    """,
                    (query_sha, args.source, len(candidates), inserted, json.dumps({"query": args.query[:240], "script": "scripts/simplemem_candidate_index.py"})),
                )
            conn.commit()
    report = {
        "action": "index",
        "execute_performed": bool(args.execute),
        "db_writes_performed": bool(args.execute),
        "graph_writes_performed": False,
        "query_sha256": query_sha,
        "source": args.source,
        "candidates_seen": len(candidates),
        "candidates_inserted": inserted,
        "safe_to_answer_from_this_alone": False,
        "not_truth": True,
        "promotion_allowed": False,
        "sample_candidates": candidates[:10],
        "blockers": [] if candidates else ["no_candidates_matched_query"],
    }
    write_report("index_execute" if args.execute else "index_dry_run", report)
    print(f"CANDIDATES_SEEN={len(candidates)}")
    print(f"CANDIDATES_INSERTED={inserted if args.execute else 0}")
    return 0 if candidates or not args.require_candidates else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Index SimpleMem recall candidates as NOT_TRUTH")
    p.add_argument("--state-database-url")
    p.add_argument("--storage-database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("index")
    sp.add_argument("--execute", action="store_true")
    sp.add_argument("--query", required=True)
    sp.add_argument("--source", choices=["claim_packets", "design_atoms", "all"], default="all")
    sp.add_argument("--limit", type=int, default=25)
    sp.add_argument("--include-zero-score", action="store_true")
    sp.add_argument("--require-candidates", action="store_true")
    sp.set_defaults(func=index)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
