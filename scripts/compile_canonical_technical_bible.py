#!/usr/bin/env python3
"""Compile DB-coordinate canon nodes into readable manuals.

PostgREST remains the API surface. This script only fetches JSON rows and renders
human-readable Markdown receipts on demand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
import psycopg
from psycopg.rows import dict_row

import scripts.root_rotor_postgrest_control as control

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL_IDS = ["SYSTEM_ARCH", "RUNTIME_GOVERNOR", "AVIONICS", "FLIGHT_MAN", "LEDGER"]
DEFAULT_OUTPUT_DIR = ROOT / "05_OUTPUTS" / "root_rotor_manuals"
DEFAULT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "root_rotor_manuals"

try:
    from jinja2 import Template
except Exception:  # pragma: no cover - only used if local env lacks Jinja2
    Template = None  # type: ignore[assignment]

TEMPLATE_MARKDOWN = """
# {{ manual_title }}
Effective Manual Version: v{{ current_version }}
---
{% for node in nodes %}
## {{ node.node_id }} {{ node.title }}
* Status: {{ node.status }} | Version: v{{ node.version }}
* Hash: {{ node.hash_current[:10] }} | Prev: {{ node.previous_hash[:10] if node.previous_hash else 'NONE' }}
{% if node.source_refs %}* Source Refs: {{ node.source_refs | join(', ') }}{% endif %}

### Specification
{{ node.payload }}

{% if node.dependencies %}* Dependencies: {{ node.dependencies | join(', ') }}{% endif %}
{% if node.affects_nodes %}* Blast Radius Impact: {{ node.affects_nodes | join(', ') }}{% endif %}
---
{% endfor %}
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manual_title(manual_id: str) -> str:
    return manual_id.replace("_", " ").title()


def sorted_nodes(nodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(node: dict[str, Any]) -> tuple[Any, str]:
        sort_key = node.get("node_sort_key")
        if isinstance(sort_key, list):
            return (sort_key, str(node.get("node_id", "")))
        return ([int(p) if p.isdigit() else 0 for p in str(node.get("node_id", "")).split(".")], str(node.get("node_id", "")))

    return sorted((dict(n) for n in nodes), key=key)


def render_manual(manual_id: str, nodes: list[dict[str, Any]]) -> str:
    ordered = sorted_nodes(nodes)
    current_version = max((int(n.get("version") or 1) for n in ordered), default=1)
    if Template is not None:
        return Template(TEMPLATE_MARKDOWN).render(
            manual_title=manual_title(manual_id),
            current_version=current_version,
            nodes=ordered,
        ).strip() + "\n"

    lines = [f"# {manual_title(manual_id)}", f"Effective Manual Version: v{current_version}", "---"]
    for node in ordered:
        lines.extend([
            f"## {node['node_id']} {node['title']}",
            f"* Status: {node['status']} | Version: v{node['version']}",
            f"* Hash: {node['hash_current'][:10]} | Prev: {node['previous_hash'][:10] if node.get('previous_hash') else 'NONE'}",
        ])
        if node.get("source_refs"):
            lines.append("* Source Refs: " + ", ".join(node["source_refs"]))
        lines.extend(["", "### Specification", str(node["payload"]), ""])
        if node.get("dependencies"):
            lines.append("* Dependencies: " + ", ".join(node["dependencies"]))
        if node.get("affects_nodes"):
            lines.append("* Blast Radius Impact: " + ", ".join(node["affects_nodes"]))
        lines.append("---")
    return "\n".join(lines).strip() + "\n"


def fetch_nodes(postgrest_url: str, manual_id: str, timeout: float) -> list[dict[str, Any]]:
    base = postgrest_url.rstrip("/")
    url = f"{base}/api_bible_nodes?manual_id=eq.{manual_id}&order=node_sort_key.asc"
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"PostgREST fetch failed for {manual_id}: {response.status_code} {response.text[:200]}")
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"PostgREST returned non-list payload for {manual_id}")
    return [dict(item) for item in payload]


def fetch_nodes_db(dsn: str, manual_id: str) -> list[dict[str, Any]]:
    sql = """
        SELECT *
        FROM lucidota_canon.api_bible_nodes
        WHERE manual_id = %s
        ORDER BY node_sort_key ASC
    """
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (manual_id,))
            return [dict(row) for row in cur.fetchall()]


def _compile_from_fetcher(
    *,
    source: str,
    fetcher: Any,
    manual_ids: list[str],
    output_dir: Path,
    receipt_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    compiled: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    generated_at = now()

    for manual_id in manual_ids:
        try:
            nodes = fetcher(manual_id)
            if not nodes:
                continue
            rendered = render_manual(manual_id, nodes)
            out = output_dir / f"compiled_{manual_id.lower()}.md"
            out.write_text(rendered, encoding="utf-8")
            compiled.append({
                "manual_id": manual_id,
                "path": str(out),
                "node_count": len(nodes),
                "sha256": sha256_text(rendered),
            })
        except Exception as exc:
            errors.append({"manual_id": manual_id, "error": str(exc)})

    result: dict[str, Any] = {
        "schema": "lucidota.root_rotor.compile_manuals.v1",
        "generated_at": generated_at,
        "source": source,
        "manuals_requested": manual_ids,
        "manuals_compiled": len(compiled),
        "manuals": compiled,
        "errors": errors,
        "status": "PASS" if compiled and not errors else ("PARTIAL" if compiled else "FAIL"),
    }
    receipt_path = receipt_dir / f"compile_manuals_{generated_at.replace(':', '').replace('-', '')}.json"
    result["receipt_path"] = str(receipt_path)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
    return result


def compile_manuals_db(
    *,
    dsn: str,
    manual_ids: list[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict[str, Any]:
    manual_ids = manual_ids or DEFAULT_MANUAL_IDS
    return _compile_from_fetcher(
        source="postgres",
        fetcher=lambda manual_id: fetch_nodes_db(dsn, manual_id),
        manual_ids=manual_ids,
        output_dir=output_dir,
        receipt_dir=receipt_dir,
    )


def compile_manuals(
    *,
    postgrest_url: str,
    manual_ids: list[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
    timeout: float = 2.0,
    require_readiness: bool = False,
) -> dict[str, Any]:
    manual_ids = manual_ids or DEFAULT_MANUAL_IDS
    if require_readiness:
        readiness = control.wait_for_readiness(timeout_seconds=timeout)
        if not readiness.get("ready"):
            receipt_dir.mkdir(parents=True, exist_ok=True)
            generated_at = now()
            result: dict[str, Any] = {
                "schema": "lucidota.root_rotor.compile_manuals.v1",
                "generated_at": generated_at,
                "source": "postgrest",
                "manuals_requested": manual_ids,
                "manuals_compiled": 0,
                "manuals": [],
                "errors": [{"manual_id": "*", "error": "postgrest_readiness_blocked", "readiness": readiness}],
                "status": "FAIL",
                "postgrest_url": postgrest_url.rstrip("/"),
            }
            receipt_path = receipt_dir / f"compile_manuals_{generated_at.replace(':', '').replace('-', '')}.json"
            result["receipt_path"] = str(receipt_path)
            receipt_path.write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
            return result
    result = _compile_from_fetcher(
        source="postgrest",
        fetcher=lambda manual_id: fetch_nodes(postgrest_url, manual_id, timeout),
        manual_ids=manual_ids,
        output_dir=output_dir,
        receipt_dir=receipt_dir,
    )
    result["postgrest_url"] = postgrest_url.rstrip("/")
    result["receipt_path"] = str(Path(result["receipt_path"]))
    Path(result["receipt_path"]).write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Canonical Technical Bible manuals from PostgREST JSON.")
    parser.add_argument("--postgrest-url", default="http://localhost:3000")
    parser.add_argument("--dsn")
    parser.add_argument("--manual-id", action="append", dest="manual_ids")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_DIR))
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--require-readiness", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.dsn:
        result = compile_manuals_db(
            dsn=args.dsn,
            manual_ids=args.manual_ids,
            output_dir=Path(args.output_dir),
            receipt_dir=Path(args.receipt_dir),
        )
    else:
        result = compile_manuals(
            postgrest_url=args.postgrest_url,
            manual_ids=args.manual_ids,
            output_dir=Path(args.output_dir),
            receipt_dir=Path(args.receipt_dir),
            timeout=args.timeout,
            require_readiness=args.require_readiness,
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"COMPILE_MANUALS={result['status']}")
        print(f"MANUALS_COMPILED={result['manuals_compiled']}")
        print(f"RECEIPT={result['receipt_path']}")
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
