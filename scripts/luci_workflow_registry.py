#!/usr/bin/env python3
"""Render live API workflow registry packet from PostgREST."""
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
    lines = ["WORKFLOW REGISTRY CURRENT"]
    if not rows:
        lines.append("- no workflow registry packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    active_count = sum(1 for item in rows if item.get("status") == "active")
    deprecated_count = sum(1 for item in rows if item.get("status") == "deprecated")
    lines.append(
        f"- rows={len(rows)} active={active_count} deprecated={deprecated_count} "
        f"first={row.get('workflow_name') or row.get('workflow_id')}"
    )
    lines.append(
        f"  first_row: workflow_id={row.get('workflow_id')} verb={row.get('verb')} "
        f"owner={row.get('owner')} phase={row.get('phase')} status={row.get('status')}"
    )
    notes = row.get("notes")
    if notes:
        lines.append(f"  notes={notes}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live API workflow registry packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["registry"], default="registry")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"api_workflow_registry?order=updated_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
