#!/usr/bin/env python3
"""Render live Indy queue packet from PostgREST."""
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
    lines = ["INDY QUEUE"]
    if not rows:
        lines.append("- no queued Indy dialogue rows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(
        f"- rows={len(rows)} first_id={row.get('id') or row.get('event_id')} "
        f"status={row.get('processed_status')} room={row.get('room_id')}"
    )
    lines.append(f"  sender={row.get('sender_id')} receipt={row.get('receipt_id')} received_at={row.get('received_at')}")
    if row.get("clean_text"):
        lines.append(f"  clean_text={str(row.get('clean_text'))[:140]}")
    if row.get("raw_text"):
        lines.append(f"  raw_text={str(row.get('raw_text'))[:140]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live Indy queue packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["queue"], default="queue")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"indy_queue?order=received_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
