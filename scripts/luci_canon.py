#!/usr/bin/env python3
"""Render live canon current packet from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["CANON CURRENT"]
    if not rows:
        lines.append("- no canon packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(
        f"- node {row.get('node_id')} :: {row.get('title')} [{row.get('status')}] version={row.get('version')}"
    )
    lines.append(f"  manual_id={row.get('manual_id')} parent_id={row.get('parent_id')} sort_key={row.get('node_sort_key')}")
    tags = row.get("ontology_tags") or []
    if tags:
        lines.append(f"  ontology_tags={', '.join(str(tag) for tag in tags)}")
    if row.get("hash_current"):
        lines.append(f"  hash_current={row.get('hash_current')}")
    if row.get("updated_at"):
        lines.append(f"  updated_at={row.get('updated_at')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live canon current packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"canon_current?order=updated_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
