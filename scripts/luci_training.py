#!/usr/bin/env python3
"""Render live training job packets from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, route: str, limit: int = 5) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode({"limit": str(limit)}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{route.lstrip('/')}?{qs}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["TRAINING JOB"]
    if not rows:
        lines.append("- no rows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(f"- rows={len(rows)} first_keys={sorted(row.keys())[:10]}")
    for key in sorted(row.keys())[:6]:
        value = row.get(key)
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"  {key}={value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live training job packets from PostgREST.")
    ap.add_argument("mode", choices=["job"])
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, "training_job", limit=args.limit)
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
