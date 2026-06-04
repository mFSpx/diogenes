#!/usr/bin/env python3
"""Render live canonical bible-edge rows from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return url, payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["BIBLE EDGES"]
    if not rows:
        lines.append("- no bible edges yet")
        return "\n".join(lines) + "\n"
    lines.append(f"- count={len(rows)}")
    for row in rows[:8]:
        lines.append(f"- {row.get('edge_id')} :: {row.get('from_node_id')} -> {row.get('to_node_id')} [{row.get('edge_kind')}]")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live bible-edge rows from PostgREST.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    url, rows = fetch_json(args.base_url, "api_bible_edges", {"order": "edge_id.asc", "limit": str(args.limit)})
    if args.json:
        print(json.dumps({"source_url": url, "payload": rows}, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
