#!/usr/bin/env python3
"""Render the live API route catalog from PostgREST."""
from __future__ import annotations

import argparse
import json
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
    lines = ["API ROUTE CATALOG"]
    if not rows:
        lines.append("- no route catalog rows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(f"- rows={len(rows)} first_route_id={row.get('route_id')} path={row.get('path_pattern')} status={row.get('status')}")
    lines.append(f"  target={row.get('target')} description={row.get('description')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show the live API route catalog from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["catalog"], default="catalog")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"api_route_catalog?order=route_id.asc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
