#!/usr/bin/env python3
"""Render live chrono current packet from PostgREST."""
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
    lines = ["CHRONO CURRENT"]
    if not rows:
        lines.append("- no chrono packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(f"- {row.get('chrono_packet_id')} refreshed={row.get('refreshed_at')}")
    for section in ["prompt_ledger", "work_ledger", "execution_history", "learning_loop", "routing_registry"]:
        value = row.get(section) or {}
        if isinstance(value, dict):
            lines.append(f"  {section}:")
            for key, val in value.items():
                lines.append(f"    {key}: {val}")
    notes = row.get("inspection_notes") or {}
    if notes:
        lines.append("  inspection_notes:")
        for key, val in notes.items():
            lines.append(f"    {key}: {val}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live chrono packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = fetch_json(args.base_url, "chrono_current?limit=5")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
