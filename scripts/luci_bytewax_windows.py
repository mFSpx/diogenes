#!/usr/bin/env python3
"""Render live Bytewax compact windows from PostgREST."""
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


def render(rows: list[dict[str, Any]], *, raw: bool = False) -> str:
    lines = [f"BYTEWAX {'RAW ' if raw else ''}WINDOWS".strip()]
    if not rows:
        lines.append("- no compact windows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(
        f"- rows={len(rows)} window={row.get('window_kind')} source={row.get('source')} "
        f"topic={row.get('topic')} object_type={row.get('object_type')}"
    )
    lines.append(
        f"  event_count={row.get('event_count')} dropped_raw_bodies={row.get('dropped_raw_bodies')} "
        f"needs_cloud_reasoning={row.get('needs_cloud_reasoning')}"
    )
    lines.append(
        f"  work_order_uuid={row.get('work_order_uuid')} window_start={row.get('window_start_at')} window_end={row.get('window_end_at')}"
    )
    if row.get("summary"):
        lines.append(f"  summary={row.get('summary')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live Bytewax compact windows from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["raw", "windows"], default="windows")
    ap.add_argument("raw_mode", nargs="?", choices=["windows"])
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    raw = args.mode == "raw"
    mode = args.raw_mode if raw else args.mode
    if raw and not mode:
        ap.error("raw requires windows")
    rows = fetch_json(args.base_url, f"bytewax_compact_windows?order=updated_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows, raw=raw), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
