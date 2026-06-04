#!/usr/bin/env python3
"""Render the live ontology todo queue from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/{path.lstrip('/')}", timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["TODO CURRENT"]
    if not rows:
        lines.append("- no ontology batches yet")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(
            f"- {row.get('batch_key')} :: {row.get('objective_summary')} "
            f"[items={row.get('item_count')} parallel={row.get('parallel_item_count')} serialized={row.get('serialized_item_count')}]"
        )
        missing = row.get("missing_executor_roles") or []
        if missing:
            lines.append(f"  missing roles: {', '.join(missing)}")
        lanes = row.get("selected_lanes") or []
        if lanes:
            lines.append("  selected lanes:")
            for lane in lanes[:8]:
                lines.append(f"  - {lane.get('role')} -> {lane.get('model_id')} ({lane.get('status')})")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live ontology todo batches from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"todo_current?limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
