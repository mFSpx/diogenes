#!/usr/bin/env python3
"""Render live daemon status packet from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.request
import urllib.parse
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["DAEMON STATUS"]
    if not rows:
        lines.append("- no daemon status packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(
        f"- daemon={row.get('daemon_name') or row.get('service_name') or row.get('name')} "
        f"heartbeat={row.get('heartbeat_kind')} pid={row.get('process_id')}"
    )
    goal = row.get("goal")
    if isinstance(goal, dict) and goal:
        lines.append(
            f"  goal={goal.get('title') or goal.get('goal_id')} status={goal.get('status')}"
        )
    db_law = row.get("db_law")
    if isinstance(db_law, dict) and db_law.get("statement"):
        lines.append(f"  db_law={db_law['statement']}")
    lines.append(
        f"  host={row.get('host_name')} socket_active={row.get('socket_active')} "
        f"terminal_active={row.get('terminal_active')} batch_size={row.get('batch_size')}"
    )
    next_commands = row.get("next_commands")
    if isinstance(next_commands, list) and next_commands:
        lines.append(f"  next_commands={len(next_commands)}")
    detail = row.get("detail") or {}
    if isinstance(detail, dict) and detail:
        lines.append("  detail:")
        for key, value in detail.items():
            lines.append(f"    {key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live daemon status packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["status"], default="status")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"daemon_status?limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
