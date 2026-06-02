#!/usr/bin/env python3
"""Deterministic Red-Team audit for Root-Rotor canon DB/manual readiness."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import requests
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "root_rotor_manuals"
DEFAULT_POSTGREST_URL = "http://127.0.0.1:3000"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_findings(metrics: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if int(metrics.get("draft_nodes") or 0) > 0:
        blockers.append("manual_incomplete_draft_nodes")
    if int(metrics.get("broken_parent_count") or 0) > 0:
        blockers.append("broken_parent_links")
    if int(metrics.get("parent_cycle_count") or 0) > 0:
        blockers.append("parent_cycle_detected")
    if int(metrics.get("dependency_cycle_count") or 0) > 0:
        blockers.append("dependency_cycle_detected")
    if not bool(metrics.get("postgrest_available")):
        blockers.append("postgrest_binary_missing")
    if not bool(metrics.get("postgrest_api_available")):
        blockers.append("postgrest_api_unavailable")
    if int(metrics.get("model_payload_count") or 0) < int(metrics.get("total_nodes") or 0):
        warnings.append("not_all_nodes_have_model_payloads")
    return {
        "verdict": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "warnings": warnings,
    }


def postgrest_api_available(postgrest_url: str, *, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(f"{postgrest_url.rstrip('/')}/api_bible_manuals?limit=1", timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code == 200


def collect_db_metrics(*, dsn: str, postgrest_url: str = DEFAULT_POSTGREST_URL) -> dict[str, Any]:
    queries = {
        "total_nodes": "SELECT count(*)::int FROM lucidota_canon.bible_nodes",
        "draft_nodes": "SELECT count(*)::int FROM lucidota_canon.bible_nodes WHERE status = 'draft'",
        "verified_nodes": "SELECT count(*)::int FROM lucidota_canon.bible_nodes WHERE status = 'verified'",
        "history_rows": "SELECT count(*)::int FROM lucidota_canon.bible_history",
        "dependency_edges": "SELECT count(*)::int FROM lucidota_canon.bible_dependencies",
        "broken_parent_count": """
            SELECT count(*)::int
            FROM lucidota_canon.bible_nodes child
            LEFT JOIN lucidota_canon.bible_nodes parent ON parent.node_id = child.parent_id
            WHERE child.parent_id IS NOT NULL AND parent.node_id IS NULL
        """,
        "model_payload_count": """
            SELECT count(*)::int
            FROM lucidota_canon.bible_nodes
            WHERE payload_format = 'json'
              AND payload::jsonb->>'schema' = 'lucidota.root_rotor.bible_node_payload.v1'
              AND payload::jsonb ? 'what_it_is_and_does'
        """,
        "parent_cycle_count": """
            WITH RECURSIVE walk(origin, node_id, parent_id, path, cycle) AS (
                SELECT node_id, node_id, parent_id, ARRAY[node_id], false
                FROM lucidota_canon.bible_nodes
              UNION ALL
                SELECT walk.origin, parent.node_id, parent.parent_id, walk.path || parent.node_id,
                       parent.node_id = ANY(walk.path)
                FROM walk
                JOIN lucidota_canon.bible_nodes parent ON parent.node_id = walk.parent_id
                WHERE NOT walk.cycle AND array_length(walk.path, 1) < 128
            )
            SELECT count(DISTINCT origin)::int FROM walk WHERE cycle
        """,
        "dependency_cycle_count": """
            WITH RECURSIVE walk(origin, node_id, path, cycle) AS (
                SELECT from_node_id, to_node_id, ARRAY[from_node_id, to_node_id], false
                FROM lucidota_canon.bible_dependencies
                WHERE edge_kind IN ('depends_on', 'affects')
              UNION ALL
                SELECT walk.origin, edge.to_node_id, walk.path || edge.to_node_id,
                       edge.to_node_id = ANY(walk.path)
                FROM walk
                JOIN lucidota_canon.bible_dependencies edge ON edge.from_node_id = walk.node_id
                WHERE NOT walk.cycle AND array_length(walk.path, 1) < 128
            )
            SELECT count(DISTINCT origin)::int FROM walk WHERE cycle
        """,
    }
    metrics: dict[str, Any] = {}
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for name, sql in queries.items():
                cur.execute(sql)
                row = cur.fetchone()
                metrics[name] = list(row.values())[0] if row else 0
            cur.execute("""
                SELECT manual_id, status, count(*)::int AS count
                FROM lucidota_canon.bible_nodes
                GROUP BY manual_id, status
                ORDER BY manual_id, status
            """)
            metrics["status_by_manual"] = [dict(row) for row in cur.fetchall()]
    metrics["postgrest_available"] = shutil.which("postgrest") is not None
    metrics["postgrest_api_available"] = postgrest_api_available(postgrest_url)
    metrics["postgrest_url"] = postgrest_url.rstrip("/")
    return metrics


def run_audit(*, dsn: str, receipt_dir: Path = DEFAULT_RECEIPT_DIR, postgrest_url: str = DEFAULT_POSTGREST_URL) -> dict[str, Any]:
    metrics = collect_db_metrics(dsn=dsn, postgrest_url=postgrest_url)
    classified = classify_findings(metrics)
    result = {
        "schema": "lucidota.root_rotor.red_team_audit.v1",
        "generated_at": now(),
        "dsn": dsn,
        "metrics": metrics,
        **classified,
    }
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"red_team_audit_{result['generated_at'].replace(':', '').replace('-', '')}.json"
    result["receipt_path"] = str(receipt_path)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Root-Rotor red-team audit checks.")
    parser.add_argument("--dsn", default="postgresql:///lucidota_state")
    parser.add_argument("--postgrest-url", default=DEFAULT_POSTGREST_URL)
    parser.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_audit(dsn=args.dsn, receipt_dir=Path(args.receipt_dir), postgrest_url=args.postgrest_url)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"ROOT_ROTOR_RED_TEAM={result['verdict']}")
        print(f"BLOCKERS={','.join(result['blockers'])}")
        print(f"RECEIPT={result['receipt_path']}")
    return 0 if result["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
