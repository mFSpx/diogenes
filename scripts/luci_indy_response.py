#!/usr/bin/env python3
"""Render live Indy response packet from PostgREST."""
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
    lines = ["INDY RESPONSES"]
    if not rows:
        lines.append("- no Indy response rows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    response_id = row.get("response_id") or row.get("id") or row.get("event_id")
    status = row.get("status") or row.get("processed_status") or row.get("delivery_status")
    lines.append(
        f"- rows={len(rows)} first_id={response_id} status={status} "
        f"created_at={row.get('created_at') or row.get('received_at')}"
    )
    if row.get("room_id"):
        lines.append(f"  room={row.get('room_id')} sender={row.get('sender_id')}")
    if row.get("body"):
        lines.append(f"  body={str(row.get('body'))[:160]}")
    elif row.get("response_body"):
        lines.append(f"  response_body={str(row.get('response_body'))[:160]}")
    elif row.get("clean_text"):
        lines.append(f"  clean_text={str(row.get('clean_text'))[:160]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live Indy response packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["responses"], default="responses")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"indy_responses?order=created_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
