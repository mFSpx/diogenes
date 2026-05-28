#!/usr/bin/env python3
"""Fast CLI navigation over LUCIDOTA's Postgres graph.

Read-only by default. This is a thin, deterministic cockpit for graph roots,
metric claims, edges, and evidence pointers; materialization stays in the
existing graph promotion/helper path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "graph_nav"
ROOT_SCOPES = ("case_corpus_root_meta_truth_only", "project_corpus_root_meta_truth_only", "system_corpus_root_meta_truth_only", "reingest_batch_root_meta_truth_only", "hunch_audit_root_meta_truth_only")

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

def db_url(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"

def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write_report(report: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"lucidota_graph_nav_{report.get('mode', 'query')}_{stamp()}.json"
    report.setdefault("generated_at_utc", now())
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}", file=sys.stderr)
    return path


def build_search_sql(query: str, limit: int, term: str | None = None) -> tuple[str, list[Any]]:
    where = []
    params: list[Any] = []
    if term:
        where.append("term=%s")
        params.append(term)
    where.append("(label ILIKE %s OR payload::text ILIKE %s)")
    like = f"%{query}%"
    params.extend([like, like, limit])
    sql = """
        SELECT uuid::text, term, status, label, payload->>'truth_scope' AS truth_scope,
               payload->'evidence_counts' AS evidence_counts, created_at::text
        FROM lucidota_go.graph_item
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT %s
    """.format(where=" AND ".join(where))
    return sql, params


def query_rows(database_url: str, sql: str, params: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]


def roots_report(database_url: str, limit: int) -> dict[str, Any]:
    sql = """
        WITH roots AS (
          SELECT uuid, uuid::text AS uuid_text, term, status, label,
                 payload->>'truth_scope' AS truth_scope,
                 payload->'evidence_counts' AS evidence_counts,
                 created_at
          FROM lucidota_go.graph_item
          WHERE payload->>'truth_scope' = ANY(%s)
          ORDER BY created_at DESC
          LIMIT %s
        )
        SELECT r.uuid_text AS uuid, r.term, r.status, r.label, r.truth_scope,
               r.evidence_counts, r.created_at::text,
               count(e.edge_uuid) FILTER (WHERE e.edge_type='HAS_METRIC_CLAIM')::int AS metric_edges,
               count(e.edge_uuid)::int AS outgoing_edges
        FROM roots r
        LEFT JOIN lucidota_go.graph_edge e ON e.source_uuid = r.uuid
        GROUP BY r.uuid, r.uuid_text, r.term, r.status, r.label, r.truth_scope, r.evidence_counts, r.created_at
        ORDER BY r.created_at DESC
    """
    rows = query_rows(database_url, sql, [list(ROOT_SCOPES), limit])
    return {"mode": "roots", "rows": rows, "root_scopes": list(ROOT_SCOPES), "status": "PASS"}


def metrics_report(database_url: str, limit: int) -> dict[str, Any]:
    sql = """
        SELECT m.uuid::text, m.term, m.status, m.label,
               m.payload->>'truth_scope' AS truth_scope,
               m.payload->'evidence_counts' AS evidence_counts,
               m.payload->>'root_graph_item_uuid' AS root_uuid,
               r.label AS root_label,
               m.created_at::text
        FROM lucidota_go.graph_item m
        LEFT JOIN lucidota_go.graph_item r
          ON r.uuid = NULLIF(m.payload->>'root_graph_item_uuid','')::uuid
        WHERE m.payload->>'truth_scope' = 'deterministic_metric_claim_from_receipts'
        ORDER BY m.created_at DESC
        LIMIT %s
    """
    return {"mode": "metrics", "rows": query_rows(database_url, sql, [limit]), "status": "PASS"}


def hunch_report(database_url: str, limit: int) -> dict[str, Any]:
    sql = """
        SELECT uuid::text, term, status, label, payload->>'truth_scope' AS truth_scope,
               payload->'evidence_counts' AS evidence_counts, created_at::text
        FROM lucidota_go.graph_item
        WHERE term='HUNCH' OR label ILIKE %s OR payload::text ILIKE %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    return {"mode": "hunch", "rows": query_rows(database_url, sql, ["%hunch%", "%hunch_hypertimeline%", limit]), "status": "PASS"}


def search_report(database_url: str, query: str, limit: int, term: str | None) -> dict[str, Any]:
    sql, params = build_search_sql(query, limit, term)
    return {"mode": "search", "query": query, "term": term, "rows": query_rows(database_url, sql, params), "status": "PASS"}


def build_chrono_sql(query: str | None, since: str | None, until: str | None, limit: int) -> tuple[str, list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if since:
        where.append("resolved_timestamp >= %s::timestamptz")
        params.append(since)
    if until:
        where.append("resolved_timestamp <= %s::timestamptz")
        params.append(until)
    if query:
        where.append("(evidence_source ILIKE %s OR source_sha256 ILIKE %s OR detail::text ILIKE %s)")
        like = f"%{query}%"
        params.extend([like, like, like])
    sql = """
        SELECT file_uuid::text, selected_claim_uuid::text, resolved_timestamp::text,
               evidence_source, trust_weight::text, source_sha256,
               projection_refreshed_at::text, ranking_method, detail
        FROM lucidota_korpus.current_chrono_timeline_projection
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY resolved_timestamp DESC LIMIT %s"
    params.append(limit)
    return sql, params


def chrono_report(database_url: str, query: str | None, since: str | None, until: str | None, limit: int) -> dict[str, Any]:
    sql, params = build_chrono_sql(query, since, until, limit)
    return {
        "mode": "chrono",
        "query": query,
        "since": since,
        "until": until,
        "rows": query_rows(database_url, sql, params),
        "status": "PASS",
    }


def build_day_chrono_sql(day: str, limit: int) -> tuple[str, list[Any]]:
    sql = """
        SELECT file_uuid::text, selected_claim_uuid::text, resolved_timestamp::text,
               evidence_source, trust_weight::text, source_sha256,
               projection_refreshed_at::text, ranking_method, detail
        FROM lucidota_korpus.current_chrono_timeline_projection
        WHERE resolved_timestamp >= %s::date
          AND resolved_timestamp < (%s::date + interval '1 day')
        ORDER BY resolved_timestamp DESC
        LIMIT %s
    """
    return sql, [day, day, limit]


def day_report(database_url: str, day: str, limit: int) -> dict[str, Any]:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uuid::text, term, status, label, payload,
                       payload->>'truth_scope' AS truth_scope,
                       payload->'evidence_counts' AS evidence_counts,
                       created_at::text
                FROM lucidota_go.graph_item
                WHERE payload->>'truth_scope'='chrono_day_bucket_meta_truth_only'
                  AND payload->>'bucket_day'=%s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (day,),
            )
            item = cur.fetchone()
            if not item:
                return {"mode": "day", "day": day, "status": "NO_ROWS", "rows": [], "edges": [], "chrono_rows": []}
            item = dict(item)
            cur.execute(
                """
                SELECT e.edge_uuid::text, e.edge_type, e.status, e.created_at::text,
                       e.source_uuid::text, s.label AS source_label,
                       e.target_uuid::text, t.label AS target_label,
                       e.detail
                FROM lucidota_go.graph_edge e
                JOIN lucidota_go.graph_item s ON s.uuid=e.source_uuid
                JOIN lucidota_go.graph_item t ON t.uuid=e.target_uuid
                WHERE e.source_uuid=%s::uuid OR e.target_uuid=%s::uuid
                ORDER BY e.created_at DESC
                LIMIT %s
                """,
                (item["uuid"], item["uuid"], limit),
            )
            edges = [dict(r) for r in cur.fetchall()]
            sql, params = build_day_chrono_sql(day, limit)
            cur.execute(sql, tuple(params))
            chrono_rows = [dict(r) for r in cur.fetchall()]
    return {"mode": "day", "day": day, "status": "PASS", "rows": [item], "edges": edges, "chrono_rows": chrono_rows}


def get_day_bucket_uuid(database_url: str, day: str) -> str | None:
    rows = query_rows(
        database_url,
        """
        SELECT uuid::text
        FROM lucidota_go.graph_item
        WHERE payload->>'truth_scope'='chrono_day_bucket_meta_truth_only'
          AND payload->>'bucket_day'=%s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [day],
    )
    return rows[0]["uuid"] if rows else None


def day_evidence_report(database_url: str, day: str, limit: int) -> dict[str, Any]:
    uuid_text = get_day_bucket_uuid(database_url, day)
    if not uuid_text:
        return {"mode": "day-evidence", "day": day, "status": "NO_BUCKET", "rows": []}
    report = evidence_report(database_url, uuid_text, limit)
    report.update({"mode": "day-evidence", "day": day, "resolved_day_uuid": uuid_text})
    return report


def resolve_item(cur: Any, target: str) -> dict[str, Any] | None:
    try:
        cur.execute(
            """
            SELECT uuid::text, term, status, label, payload,
                   payload->>'truth_scope' AS truth_scope,
                   payload->'evidence_counts' AS evidence_counts,
                   created_at::text
            FROM lucidota_go.graph_item WHERE uuid=%s::uuid
            """,
            (target,),
        )
    except Exception:
        cur.connection.rollback()
        cur.execute(
            """
            SELECT uuid::text, term, status, label, payload,
                   payload->>'truth_scope' AS truth_scope,
                   payload->'evidence_counts' AS evidence_counts,
                   created_at::text
            FROM lucidota_go.graph_item
            WHERE label ILIKE %s OR payload::text ILIKE %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (f"%{target}%", f"%{target}%"),
        )
    row = cur.fetchone()
    return dict(row) if row else None


def evidence_report(database_url: str, target: str, limit: int) -> dict[str, Any]:
    refs: dict[tuple[str, str], int] = {}

    def add(source: str, values: Any) -> None:
        if values is None:
            return
        if isinstance(values, str):
            values = [values]
        if isinstance(values, dict):
            values = values.values()
        for value in values:
            if isinstance(value, (list, tuple, set)):
                add(source, list(value))
                continue
            ref = str(value).strip()
            if not ref:
                continue
            refs[(source, ref)] = refs.get((source, ref), 0) + 1

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            item = resolve_item(cur, target)
            if not item:
                return {"mode": "evidence", "target": target, "status": "NO_ROWS", "rows": []}
            uuid_text = item["uuid"]
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            add("payload.source_receipts", payload.get("source_receipts"))
            add("payload.evidence_refs", payload.get("evidence_refs"))
            cur.execute(
                """
                SELECT evidence_refs
                FROM lucidota_go.graph_promotion_materialization
                WHERE graph_item_uuid=%s::uuid
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (uuid_text, limit),
            )
            for row in cur.fetchall():
                add("materialization", row["evidence_refs"])
            cur.execute(
                """
                SELECT detail
                FROM lucidota_go.graph_edge
                WHERE source_uuid=%s::uuid OR target_uuid=%s::uuid
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (uuid_text, uuid_text, limit),
            )
            for row in cur.fetchall():
                detail = row["detail"] if isinstance(row["detail"], dict) else {}
                add("edge.source_evidence_refs", detail.get("source_evidence_refs"))
                add("edge.target_evidence_refs", detail.get("target_evidence_refs"))
    priority = {
        "payload.source_receipts": 0,
        "payload.evidence_refs": 1,
        "materialization": 2,
        "edge.source_evidence_refs": 3,
        "edge.target_evidence_refs": 4,
    }
    rows = [
        {"source": source, "ref": ref, "count": count}
        for (source, ref), count in sorted(refs.items(), key=lambda item: (priority.get(item[0][0], 99), -item[1], item[0][1]))
    ][:limit]
    return {"mode": "evidence", "target": target, "resolved_item": item, "rows": rows, "status": "PASS"}


def show_report(database_url: str, target: str, limit: int) -> dict[str, Any]:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            item = resolve_item(cur, target)
            if not item:
                return {"mode": "show", "target": target, "status": "NO_ROWS", "rows": [], "edges": []}
            uuid_text = item["uuid"]
            cur.execute(
                """
                SELECT e.edge_uuid::text, e.edge_type, e.status, e.created_at::text,
                       e.source_uuid::text, s.label AS source_label,
                       e.target_uuid::text, t.label AS target_label,
                       e.detail
                FROM lucidota_go.graph_edge e
                JOIN lucidota_go.graph_item s ON s.uuid=e.source_uuid
                JOIN lucidota_go.graph_item t ON t.uuid=e.target_uuid
                WHERE e.source_uuid=%s::uuid OR e.target_uuid=%s::uuid
                ORDER BY e.created_at DESC
                LIMIT %s
                """,
                (uuid_text, uuid_text, limit),
            )
            edges = [dict(r) for r in cur.fetchall()]
            cur.execute(
                """
                SELECT m.materialization_uuid::text, m.materialization_kind,
                       m.evidence_refs, m.created_at::text, h.helper_receipt_uuid::text
                FROM lucidota_go.graph_promotion_materialization m
                LEFT JOIN lucidota_go.graph_materialization_helper_receipt h
                  ON h.materialization_uuid=m.materialization_uuid
                WHERE m.graph_item_uuid=%s::uuid
                ORDER BY m.created_at DESC
                LIMIT %s
                """,
                (uuid_text, limit),
            )
            materializations = [dict(r) for r in cur.fetchall()]
    return {"mode": "show", "target": target, "status": "PASS", "rows": [dict(item)], "edges": edges, "materializations": materializations}


def render_counts(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts = []
    for key in ("files_indexed", "graph_candidates_staged", "bitloops_accepted", "river_training_lane_count", "accepted_total", "unique_hunch_ids_total", "resolved_hunches", "materially_correct_count", "projected_event_count"):
        if key in value:
            parts.append(f"{key}={value[key]}")
    return " ".join(parts)


def render_report(report: dict[str, Any]) -> str:
    mode = report.get("mode", "query")
    rows = report.get("rows") or []
    lines = [f"LUCIDOTA GRAPH NAV :: {mode} :: rows={len(rows)}"]
    if report.get("status") not in (None, "PASS"):
        lines.append(f"status={report.get('status')}")
    if mode in {"evidence", "day-evidence"}:
        for row in rows:
            lines.append(f"- {row.get('source')} count={row.get('count')} :: {row.get('ref')}")
        return "\n".join(lines)
    if mode == "chrono":
        for row in rows:
            lines.append(
                f"- {row.get('resolved_timestamp')} trust={row.get('trust_weight')} "
                f"source={row.get('evidence_source')} file={row.get('file_uuid')} sha={row.get('source_sha256') or ''}"
            )
        return "\n".join(lines)
    for row in rows:
        suffix = render_counts(row.get("evidence_counts"))
        if mode == "roots":
            suffix = f"metrics={row.get('metric_edges', 0)} edges={row.get('outgoing_edges', 0)} {suffix}".strip()
        elif mode == "metrics" and row.get("root_label"):
            suffix = f"root={row.get('root_label')} {suffix}".strip()
        lines.append(f"- {row.get('term')} {row.get('uuid')} :: {row.get('label')} :: {row.get('truth_scope') or ''} :: {suffix}".rstrip())
    for edge in report.get("edges") or []:
        lines.append(f"  edge {edge.get('edge_type')} {edge.get('edge_uuid')} :: {edge.get('source_label')} -> {edge.get('target_label')}")
    if mode == "day":
        for row in report.get("chrono_rows") or []:
            lines.append(
                f"  chrono {row.get('resolved_timestamp')} trust={row.get('trust_weight')} "
                f"source={row.get('evidence_source')} file={row.get('file_uuid')} sha={row.get('source_sha256') or ''}"
            )
    for mat in report.get("materializations") or []:
        lines.append(f"  mat {mat.get('materialization_kind')} {mat.get('materialization_uuid')} helper={mat.get('helper_receipt_uuid')}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-graph-nav")
    ap.add_argument("--database-url")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-report", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    def output_flags(parser: argparse.ArgumentParser) -> None:
        # Also allow output flags after subcommands.
        parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        parser.add_argument("--write-report", action="store_true", default=argparse.SUPPRESS)

    def add_cmd(name: str, argspec: list[str] | None = None, limit: int = 20) -> None:
        p = sub.add_parser(name)
        output_flags(p)
        for spec in argspec or []:
            p.add_argument(spec)
        p.add_argument("--limit", type=int, default=limit)

    add_cmd("roots")
    add_cmd("metrics")
    add_cmd("hunch")
    add_cmd("search", ["query"])
    sub.choices["search"].add_argument("--term")
    add_cmd("show", ["target"])
    add_cmd("evidence", ["target"], 40)
    add_cmd("chrono", limit=40)
    for opt in ("--query", "--since", "--until"):
        sub.choices["chrono"].add_argument(opt)
    add_cmd("day", ["day"])
    add_cmd("day-evidence", ["day"], 40)

    args = ap.parse_args()
    command = args.cmd or "roots"
    database_url = db_url(args)
    limit = max(1, getattr(args, "limit", 20))
    reports = {
        "roots": lambda: roots_report(database_url, limit),
        "metrics": lambda: metrics_report(database_url, limit),
        "hunch": lambda: hunch_report(database_url, limit),
        "search": lambda: search_report(database_url, args.query, limit, args.term),
        "show": lambda: show_report(database_url, args.target, limit),
        "evidence": lambda: evidence_report(database_url, args.target, limit),
        "chrono": lambda: chrono_report(database_url, args.query, args.since, args.until, limit),
        "day": lambda: day_report(database_url, args.day, limit),
        "day-evidence": lambda: day_evidence_report(database_url, args.day, limit),
    }
    report = reports[command]()
    if args.write_report:
        write_report(report)
    print(json.dumps(report, sort_keys=True, default=str) if args.json else render_report(report))
    return 0 if report.get("status") == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
