#!/usr/bin/env python3
"""Render live root orchestrator packet from PostgREST."""
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
    lines = ["ROOT ORCHESTRATOR CURRENT"]
    if not rows:
        lines.append("- no root orchestrator packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(
        f"- orchestrator_id={row.get('orchestrator_id')} title={row.get('title')} "
        f"status={row.get('status')} node_count={row.get('node_count')}"
    )
    if row.get("live_surface"):
        lines.append(f"  live_surface_keys={sorted((row.get('live_surface') or {}).keys())[:12]}")
    if row.get("next_commands"):
        lines.append(f"  next_commands={len(row.get('next_commands') or [])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live root orchestrator packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=1)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"root_orchestrator_current?limit={args.limit}")
    if args.json:
        print(json.dumps({"ok": bool(rows), "rows": rows, "source_url": f"{args.base_url.rstrip('/')}/root_orchestrator_current?limit={args.limit}"}, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
