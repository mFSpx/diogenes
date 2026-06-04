#!/usr/bin/env python3
"""Render live canonical bible-node rows from PostgREST."""
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


def render(rows: list[dict[str, Any]], manual_id: str) -> str:
    lines = [f"BIBLE NODES :: {manual_id}"]
    if not rows:
        lines.append("- no bible nodes yet")
        return "\n".join(lines) + "\n"
    first = rows[0]
    lines.append(f"- count={len(rows)} first={first.get('node_id')} :: {first.get('title')} [{first.get('status')}]")
    lines.append(f"  parent_id={first.get('parent_id')} sort_key={first.get('node_sort_key')} version={first.get('version')}")
    for row in rows[1:5]:
        lines.append(f"- {row.get('node_id')} :: {row.get('title')} [{row.get('status')}]")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live bible-node rows from PostgREST.")
    ap.add_argument("--manual-id", required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    url, rows = fetch_json(
        args.base_url,
        "api_bible_nodes",
        {"manual_id": f"eq.{args.manual_id}", "order": "node_sort_key.asc", "limit": str(args.limit)},
    )
    if args.json:
        print(json.dumps({"source_url": url, "payload": rows}, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows, args.manual_id), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
