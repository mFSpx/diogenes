#!/usr/bin/env python3
"""staging_promote_worker.py — DRY-RUN promote gate for lucidota_go.staging_packet rows.

Reads pending staging_packet rows WHERE parser_name='corpus_to_graph', then for
each candidate calls graph_promotion_gate.py gate --dry-run with
authority-class=model_computed_finding.  Does NOT write canonical graph tables.
Mutation class: receipt_only (gate report files only).

Usage:
  python3 scripts/staging_promote_worker.py --limit 5
  python3 scripts/staging_promote_worker.py --limit 20 --parser-name corpus_to_graph
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT_GRAPH = ROOT / "05_OUTPUTS" / "graph"
GATE_SCRIPT = ROOT / "scripts" / "graph_promotion_gate.py"

DEFAULT_PARSER = "corpus_to_graph"
DEFAULT_AUTHORITY = "model_computed_finding"
DEFAULT_CANDIDATE_KIND = "node"
DEFAULT_DB_URL = (
    os.environ.get("LUCIDOTA_GO_STORAGE_DSN")
    or os.environ.get("KORPUS_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
    or "postgresql:///lucidota_storage"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_gate_dry_run(
    proposed_term: str,
    claim: str,
    source_id: str,
    candidate_kind: str,
    authority_class: str,
    db_url: str,
) -> dict[str, Any]:
    """Invoke graph_promotion_gate.py gate --dry-run for a single candidate."""
    candidate_payload = json.dumps(
        {"term": proposed_term, "label": claim, "evidence_note": "corpus"}
    )
    cmd = [
        sys.executable,
        str(GATE_SCRIPT),
        "gate",
        "--dry-run",
        "--candidate-kind", candidate_kind,
        "--candidate-payload-json", candidate_payload,
        "--evidence-ref", source_id,
        "--authority-class", authority_class,
    ]
    if db_url:
        cmd = [sys.executable, str(GATE_SCRIPT), "--database-url", db_url, "gate",
               "--dry-run",
               "--candidate-kind", candidate_kind,
               "--candidate-payload-json", candidate_payload,
               "--evidence-ref", source_id,
               "--authority-class", authority_class]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    report_path = None
    allowed = False
    for line in result.stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            report_path = line.split("=", 1)[1].strip()
        if line.startswith("GRAPH_GATE_ALLOWED="):
            allowed = line.split("=", 1)[1].strip().lower() == "true"

    return {
        "proposed_term": proposed_term,
        "claim": claim,
        "source_id": source_id,
        "returncode": result.returncode,
        "allowed": allowed,
        "report_path": report_path,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip()[:500] if result.stderr else "",
    }


def fetch_pending(
    conn_str: str,
    parser_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    with psycopg.connect(conn_str, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT packet_uuid, source_id, proposed_term, claim, status
                FROM lucidota_go.staging_packet
                WHERE parser_name = %s
                  AND status = 'pending'
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (parser_name, limit),
            )
            return cur.fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description="DRY-RUN promote gate for staging_packet corpus candidates.")
    ap.add_argument("--limit", type=int, default=5, help="Max candidates to process (default 5)")
    ap.add_argument("--parser-name", default=DEFAULT_PARSER, help=f"parser_name filter (default: {DEFAULT_PARSER})")
    ap.add_argument("--authority-class", default=DEFAULT_AUTHORITY)
    ap.add_argument("--candidate-kind", default=DEFAULT_CANDIDATE_KIND, choices=["node", "edge", "property", "doctrine", "workflow", "other"])
    ap.add_argument("--database-url", default=DEFAULT_DB_URL)
    ap.add_argument("--json", action="store_true", help="Print summary JSON to stdout")
    args = ap.parse_args()

    print(f"[staging_promote_worker] fetching up to {args.limit} pending rows from staging_packet WHERE parser_name='{args.parser_name}'", flush=True)

    rows = fetch_pending(args.database_url, args.parser_name, args.limit)
    if not rows:
        print("[staging_promote_worker] no pending rows found — nothing to do", flush=True)
        return 0

    print(f"[staging_promote_worker] found {len(rows)} candidates — running gate --dry-run for each", flush=True)
    OUT_GRAPH.mkdir(parents=True, exist_ok=True)

    results = []
    passed = 0
    blocked = 0
    errors = 0

    for row in rows:
        proposed_term = row.get("proposed_term") or ""
        claim = row.get("claim") or ""
        source_id = row.get("source_id") or ""
        packet_uuid = str(row.get("packet_uuid", ""))

        if not proposed_term:
            proposed_term = f"staging_{packet_uuid[:8]}"
        if not claim:
            claim = "unknown"
        if not source_id:
            source_id = f"staging_packet:{packet_uuid}"

        print(f"  -> gate dry-run: term={proposed_term!r} source={source_id!r}", flush=True)

        gate_result = run_gate_dry_run(
            proposed_term=proposed_term,
            claim=claim,
            source_id=source_id,
            candidate_kind=args.candidate_kind,
            authority_class=args.authority_class,
            db_url=args.database_url,
        )
        gate_result["packet_uuid"] = packet_uuid

        if gate_result["returncode"] not in (0, 2):
            errors += 1
            status = "error"
        elif gate_result["allowed"]:
            passed += 1
            status = "gate_passed_dry_run"
        else:
            blocked += 1
            status = "gate_blocked_dry_run"

        gate_result["gate_status"] = status
        results.append(gate_result)
        print(f"     {status}  report={gate_result['report_path'] or 'none'}", flush=True)

    summary = {
        "worker": "staging_promote_worker",
        "mutation_class": "receipt_only",
        "mode": "DRY_RUN",
        "canonical_writes": False,
        "parser_name": args.parser_name,
        "authority_class": args.authority_class,
        "candidates_fetched": len(rows),
        "gate_passed_dry_run": passed,
        "gate_blocked_dry_run": blocked,
        "errors": errors,
        "results": results,
        "generated_at": now(),
    }

    summary_path = OUT_GRAPH / f"staging_promote_worker_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[staging_promote_worker] SUMMARY_PATH={summary_path.relative_to(ROOT)}", flush=True)
    print(f"[staging_promote_worker] passed={passed} blocked={blocked} errors={errors}", flush=True)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
