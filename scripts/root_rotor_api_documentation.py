#!/usr/bin/env python3
"""Generate Root-Law API documentation (HTML + markdown) and map API endpoints to bible nodes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DSN = "postgresql:///lucidota_state"
DEFAULT_TEMPLATE = ROOT / "scripts" / "templates" / "root_law_api_docs.html.j2"
DEFAULT_OUTPUT_DIR = ROOT / "05_OUTPUTS" / "root_rotor_manuals"
DEFAULT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "root_rotor_manuals"
DEFAULT_MANUAL_IDS = ["SYSTEM_ARCH", "RUNTIME_GOVERNOR", "AVIONICS", "FLIGHT_MAN", "LEDGER"]
ENDPOINT_PARENT = "4.0.0"
ENDPOINT_MANUAL = "FLIGHT_MAN"
ENDPOINT_NODE_START = 9000


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_nodes(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, str]:
        sort_key = row.get("node_sort_key")
        if isinstance(sort_key, list):
            return (sort_key, str(row.get("node_id", "")))
        return ([(int(p) if p.isdigit() else 0) for p in str(row.get("node_id", "")).split(".")], str(row.get("node_id", "")))

    return sorted((dict(r) for r in rows), key=key)


def fetch_rows(sql: str, dsn: str, params: tuple[Any, ...] | None = None, *, all_rows: bool = True) -> list[dict[str, Any]]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall() if all_rows else [cur.fetchone()]
    return [dict(r) for r in rows if r is not None]


def fetch_manual_nodes(dsn: str, manual_ids: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    if manual_ids:
        placeholders = ",".join("%s" for _ in manual_ids)
        rows = fetch_rows(
            f"SELECT * FROM lucidota_canon.api_bible_nodes WHERE manual_id IN ({placeholders}) ORDER BY manual_id, node_sort_key",
            dsn,
            tuple(manual_ids),
        )
    else:
        rows = fetch_rows("SELECT * FROM lucidota_canon.api_bible_nodes ORDER BY manual_id, node_sort_key", dsn)
    grouped: dict[str, list[dict[str, Any]]] = {m: [] for m in (manual_ids or [])}
    for row in rows:
        grouped.setdefault(row["manual_id"], []).append(row)
        if row["manual_id"] not in grouped:
            grouped[row["manual_id"]] = [row]
    return grouped


def fetch_route_catalog(dsn: str) -> list[dict[str, Any]]:
    return fetch_rows("SELECT * FROM lucidota_canon.api_bible_route_catalog ORDER BY route_id", dsn)


def fetch_latest_audit(prefix: str, root: Path = DEFAULT_RECEIPT_DIR) -> dict[str, Any] | None:
    if not root.exists():
        return None
    candidates = sorted(root.glob(f"{prefix}_*.json"))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return read_json(newest)


def render_api_payload(manuals: dict[str, list[dict[str, Any]]], routes: list[dict[str, Any]], contradictions: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lucidota.root_law_api_payload.v1",
        "generated_at": now(),
        "manuals": [
            {"manual_id": manual_id, "nodes": sorted_nodes(nodes)}
            for manual_id, nodes in manuals.items()
        ],
        "api_routes": routes,
        "contradictions": contradictions,
    }


def render_html(template_text: str, payload: dict[str, Any]) -> str:
    contradictions = payload.get("contradictions", {})
    payload = {
        "generated_at": payload.get("generated_at", now()),
        "manuals": payload["manuals"],
        "api_routes": payload["api_routes"],
        "contradictions": payload.get("contradictions", {}),
        "blockers": contradictions.get("blockers", []),
        "warnings": contradictions.get("warnings", []),
        "coverage_ratio": contradictions.get("coverage_ratio", 0),
    }
    return Template(template_text).render(**payload)


def read_default_contradictions(*, receipt_dir: Path = DEFAULT_RECEIPT_DIR) -> dict[str, Any]:
    sidecar = fetch_latest_audit("sidecar_anomaly_audit", receipt_dir) or {}
    redteam = fetch_latest_audit("red_team_audit", receipt_dir) or {}
    return {
        "blockers": sidecar.get("blockers", []) + redteam.get("blockers", []),
        "warnings": sidecar.get("warnings", []) + redteam.get("warnings", []),
        "coverage_ratio": sidecar.get("metrics", {}).get("coverage_ratio", 0),
    }


def _parse_route_node_id(candidate: str) -> int | None:
    parts = candidate.split(".")
    if len(parts) != 3:
        return None
    try:
        if parts[0] != "4":
            return None
        return int(parts[1])
    except ValueError:
        return None


def sync_routes_to_bible_nodes(dsn: str, routes: list[dict[str, Any]], *, include_all: bool = True) -> dict[str, Any]:
    if not routes:
        return {"upserted": 0, "updated": 0, "errors": []}

    sync_sql = """
        INSERT INTO lucidota_canon.bible_nodes(
            node_id, parent_id, node_sort_key, manual_id, node_kind, title, payload, payload_format,
            ontology_tags, source_refs, evidence_hashes, dependencies, affects_nodes, status, hash_current
        ) VALUES (
            %(node_id)s, %(parent_id)s, lucidota_canon.fn_bible_node_sort_key(%(node_id)s), %(manual_id)s,
            %(node_kind)s, %(title)s, %(payload)s, %(payload_format)s,
            %(ontology_tags)s, %(source_refs)s, %(evidence_hashes)s,
            %(dependencies)s, %(affects_nodes)s, %(status)s, repeat('0', 64)
        )
        ON CONFLICT (node_id) DO UPDATE SET
            parent_id = EXCLUDED.parent_id,
            node_kind = EXCLUDED.node_kind,
            title = EXCLUDED.title,
            payload = EXCLUDED.payload,
            payload_format = EXCLUDED.payload_format,
            ontology_tags = EXCLUDED.ontology_tags,
            source_refs = EXCLUDED.source_refs,
            evidence_hashes = EXCLUDED.evidence_hashes,
            dependencies = EXCLUDED.dependencies,
            affects_nodes = EXCLUDED.affects_nodes,
            status = EXCLUDED.status,
            updated_at = now();
    """

    used_ids: set[int] = set()
    payloads = {"upserted": 0, "updated": 0, "errors": []}

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for route in routes:
                route_uri = f"api://route/{route['route_id']}"
                cur.execute(
                    "SELECT node_id FROM lucidota_canon.bible_nodes WHERE manual_id=%s AND payload_format='json' AND source_refs @> %s::jsonb LIMIT 1",
                    (ENDPOINT_MANUAL, json.dumps([route_uri])),
                )
                row = cur.fetchone()
                if row:
                    node_id = str(row[0])
                else:
                    cur.execute(
                        "SELECT node_id FROM lucidota_canon.bible_nodes WHERE manual_id=%s AND node_id LIKE '4.%'",
                        (ENDPOINT_MANUAL,),
                    )
                    for old in cur.fetchall():
                        nid = _parse_route_node_id(str(old[0]))
                        if nid is not None:
                            used_ids.add(nid)
                    n = ENDPOINT_NODE_START
                    while n in used_ids:
                        n += 1
                    node_id = f"4.{n}.0"
                    used_ids.add(n)

                payload = {
                    "schema": "lucidota.root_rotor.route_payload.v1",
                    "source": "api_bible_route_catalog",
                    "route": route,
                    "generated_at": now(),
                }
                record = {
                    "node_id": node_id,
                    "parent_id": ENDPOINT_PARENT,
                    "manual_id": ENDPOINT_MANUAL,
                    "node_kind": "REFERENCE",
                    "title": f"API route {route['route_id']}",
                    "payload": json.dumps(payload, sort_keys=True),
                    "payload_format": "json",
                    "ontology_tags": ["API", "REFERENCE", "RECEIPT"],
                    "source_refs": [route_uri],
                    "evidence_hashes": [],
                    "dependencies": [],
                    "affects_nodes": [],
                    "status": "verified" if include_all else "review_required",
                }
                try:
                    cur.execute(sync_sql, record)
                    payloads["updated"] += 1
                except Exception as exc:  # pragma: no cover - error path
                    payloads["errors"].append({"route_id": route["route_id"], "error": str(exc)})

            payloads["upserted"] = payloads["updated"]
        conn.commit()
    return payloads


def run(
    *,
    dsn: str,
    manual_ids: list[str],
    template: Path,
    output_dir: Path,
    receipt_dir: Path,
    sync_routes: bool,
    include_all: bool,
) -> dict[str, Any]:
    manual_rows = fetch_manual_nodes(dsn, manual_ids)
    routes = fetch_route_catalog(dsn)
    contradictions = read_default_contradictions(receipt_dir=receipt_dir)
    payload = render_api_payload(manual_rows, routes, contradictions)

    template_text = template.read_text(encoding="utf-8")
    html = render_html(template_text, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "root_law_api_docs.html"
    html_path.write_text(html, encoding="utf-8")

    markdown_path = output_dir / "root_law_api_docs.md"
    lines = [f"# Root-Law Technical Bible", f"Generated: {payload['generated_at']}", "", f"Manuals: {len(payload['manuals'])}", f"API routes: {len(payload['api_routes'])}"]
    for manual in payload["manuals"]:
        lines.append(f"\n## {manual['manual_id']}")
        for node in manual["nodes"]:
            lines.append(f"- {node['node_id']} {node['title']} ({node['status']} v{node['version']})")
    for route in payload["api_routes"]:
        lines.append(f"- [{route['method']}] {route['path_pattern']} -> {route['route_id']}")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    sync_result = sync_routes_to_bible_nodes(dsn, routes, include_all=include_all) if sync_routes else {"upserted": 0, "updated": 0, "errors": []}

    result: dict[str, Any] = {
        "schema": "lucidota.root_rotor.api_documentation.v1",
        "generated_at": now(),
        "manual_ids": manual_ids,
        "route_count": len(routes),
        "html_path": str(html_path),
        "markdown_path": str(markdown_path),
        "contradictions": contradictions,
        "sync_routes": sync_routes,
        "sync_routes_result": sync_result,
        "status": "PASS" if not sync_result.get("errors") else "PARTIAL",
    }

    receipt_path = receipt_dir / f"root_law_api_docs_{result['generated_at'].replace(':', '').replace('-', '')}.json"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
    result["receipt_path"] = str(receipt_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Root-Law API docs and optionally sync endpoint nodes.")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--manual-id", action="append", default=DEFAULT_MANUAL_IDS)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_DIR))
    parser.add_argument("--sync-route-nodes", action="store_true")
    parser.add_argument("--include-all", action="store_true", help="Mark synced endpoints as verified (default review_required).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(
        dsn=args.dsn,
        manual_ids=args.manual_id,
        template=Path(args.template),
        output_dir=Path(args.output_dir),
        receipt_dir=Path(args.receipt_dir),
        sync_routes=args.sync_route_nodes,
        include_all=args.include_all,
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"ROOT_LAW_API_DOCS={result['status']}")
        print(f"HTML={result['html_path']}")
        print(f"ROUTE_ROWS={result['route_count']}")
        print(f"RECEIPT={result['receipt_path']}")
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
