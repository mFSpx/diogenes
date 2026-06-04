#!/usr/bin/env python3
"""Render live capability registry current packet from PostgREST."""
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
    lines = ["CAPABILITY CURRENT"]
    if not rows:
        lines.append("- no capability registry packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    summary = row.get("capability_summary") or {}
    lines.append(
        f"- packet {row.get('capability_packet_id')} :: capabilities={summary.get('capability_count')} "
        f"active={summary.get('active_count')} groups={summary.get('group_count')} workflows={summary.get('workflow_name_count')}"
    )
    lines.append(
        f"  active_rows={len(row.get('active_capabilities') or [])} "
        f"status_keys={sorted((row.get('status_breakdown') or {}).keys())}"
    )
    workflow_names = summary.get("workflow_names") or []
    if workflow_names:
        lines.append(f"  workflow_names={', '.join(str(name) for name in workflow_names[:12])}")
    routing_notes = row.get("routing_notes") or {}
    if routing_notes:
        lines.append("  routing_notes:")
        for key, value in routing_notes.items():
            lines.append(f"    {key}: {value}")
    active_rows = row.get("active_capabilities") or []
    if active_rows:
        lines.append("  active_capabilities:")
        for cap in active_rows[:12]:
            lines.append(
                f"    - {cap.get('capability_key')} :: {cap.get('capability_name')} "
                f"[{cap.get('lifecycle_status')}/{cap.get('run_state')}] -> {cap.get('workflow_name')}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live capability registry packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"capability_current?order=refreshed_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
