#!/usr/bin/env python3
"""Render live Indy book / LoRA / training packets from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"
ROUTES = {
    "source": "book_source",
    "scan": "book_scan",
    "read-queue": "book_read_queue",
    "note": "book_note",
    "candidate": "lora_candidate",
    "adapter": "lora_adapter",
    "training": "training_job",
    "receipt": "book_receipt",
}


def fetch_json(base_url: str, route: str, limit: int = 5) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode({"limit": str(limit)}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{route.lstrip('/')}?{qs}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(mode: str, rows: list[dict[str, Any]], *, raw: bool = False) -> str:
    label = f"BOOK {'RAW ' if raw else ''}{mode.upper()}".strip()
    lines = [label]
    if not rows:
        lines.append("- no rows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(f"- rows={len(rows)} first_keys={sorted(row.keys())[:10]}")
    for key in sorted(row.keys())[:6]:
        value = row.get(key)
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"  {key}={value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live book/LoRA/training packets from PostgREST.")
    ap.add_argument("mode", choices=["raw", *sorted(ROUTES.keys())])
    ap.add_argument("raw_mode", nargs="?", choices=sorted(ROUTES.keys()))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    raw = args.mode == "raw"
    mode = args.raw_mode if raw else args.mode
    if raw and not mode:
        ap.error("raw requires one of source|scan|read-queue|note|candidate|adapter|training|receipt")
    rows = fetch_json(args.base_url, ROUTES[mode], limit=args.limit)
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(mode, rows, raw=raw), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
