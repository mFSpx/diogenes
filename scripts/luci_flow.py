#!/usr/bin/env python3
"""Render live flow specs and receipts from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"

ROUTES = {
    "specs": "flow_specs",
    "receipts": "flow_receipts",
}


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(mode: str, rows: list[dict[str, Any]]) -> str:
    title = f"FLOW {mode.upper()}"
    lines = [title]
    if not rows:
        lines.append("- no flow rows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    if mode == "specs":
        lines.append(f"- rows={len(rows)} first_flow_id={row.get('flow_id')} name={row.get('name')} status={row.get('status')}")
        lines.append(f"  receipt_id={row.get('receipt_id')} created_at={row.get('created_at')} updated_at={row.get('updated_at')}")
    else:
        lines.append(f"- rows={len(rows)} first_receipt_id={row.get('receipt_id')} flow_id={row.get('flow_id')} status={row.get('status')}")
        metrics = row.get("metrics")
        if isinstance(metrics, dict) and metrics:
            lines.append(f"  metrics={json.dumps(metrics, sort_keys=True, ensure_ascii=False)}")
        if row.get("output_path"):
            lines.append(f"  output_path={row.get('output_path')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live flow specs or flow receipts from PostgREST.")
    ap.add_argument("mode", choices=sorted(ROUTES.keys()))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    route = ROUTES[args.mode]
    rows = fetch_json(args.base_url, f"{route}?order=updated_at.desc&limit={args.limit}" if args.mode == "specs" else f"{route}?order=created_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(args.mode, rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
