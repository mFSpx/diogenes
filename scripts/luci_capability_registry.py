#!/usr/bin/env python3
"""Render live capability registry packet from PostgREST."""
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
    lines = ["CAPABILITY REGISTRY"]
    if not rows:
        lines.append("- no capability registry rows yet")
        return "\n".join(lines) + "\n"
    lines.append(f"- rows={len(rows)} active={sum(1 for r in rows if r.get('lifecycle_status') == 'active' or r.get('run_state') == 'ran')}")
    row = rows[0]
    lines.append(
        f"  first_row: capability_key={row.get('capability_key')} group={row.get('capability_group')} "
        f"status={row.get('lifecycle_status')} workflow={row.get('workflow_name')}"
    )
    detail = row.get("detail") or {}
    if isinstance(detail, dict) and detail:
        lines.append(f"  detail_keys={sorted(detail.keys())}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live capability registry packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["registry"], default="registry")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"capability_registry?order=updated_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
