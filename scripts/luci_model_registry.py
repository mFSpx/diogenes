#!/usr/bin/env python3
"""Render live model registry current packet from PostgREST."""
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
    lines = ["MODEL REGISTRY CURRENT"]
    if not rows:
        lines.append("- no model registry packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    summary = row.get("model_summary") or {}
    lines.append(
        f"- packet {row.get('model_packet_id')} :: models={summary.get('model_count')} "
        f"active={summary.get('active_count')} roles={summary.get('role_count')} loadouts={summary.get('loadout_count')}"
    )
    lines.append(
        f"  active_rows={len(row.get('active_models') or [])} "
        f"role_keys={sorted((row.get('role_breakdown') or {}).keys())}"
    )
    role_names = summary.get("role_names") or []
    if role_names:
        lines.append(f"  role_names={', '.join(str(name) for name in role_names[:12])}")
    loadout_names = summary.get("loadout_names") or []
    if loadout_names:
        lines.append(f"  loadout_names={', '.join(str(name) for name in loadout_names[:12])}")
    routing_notes = row.get("routing_notes") or {}
    if routing_notes:
        lines.append("  routing_notes:")
        for key, value in routing_notes.items():
            lines.append(f"    {key}: {value}")
    active_rows = row.get("active_models") or []
    if active_rows:
        lines.append("  active_models:")
        for model in active_rows[:12]:
            lines.append(
                f"    - {model.get('model_id')} :: {model.get('role')} "
                f"[{model.get('active')}] -> {model.get('loadout_id')}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live model registry packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"model_registry_current?order=refreshed_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
