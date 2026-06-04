#!/usr/bin/env python3
"""Render CLI payload archive status from PostgREST."""
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
    lines = ["PAYLOAD ARCHIVE STATUS"]
    if not rows:
        lines.append("- no archived payloads yet")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(
            f"- {row.get('source_table')}::{row.get('payload_kind')} count={row.get('archive_count')} "
            f"bytes={row.get('archived_bytes')} chars={row.get('archived_chars')} latest={row.get('latest_archived_at')}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show cold payload archive status from PostgREST.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"payload_archive_status?order=source_table.asc,payload_kind.asc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
