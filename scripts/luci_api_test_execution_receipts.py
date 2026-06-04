#!/usr/bin/env python3
"""Render live API test execution receipts from PostgREST."""
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
    lines = ["API TEST EXECUTION RECEIPTS"]
    if not rows:
        lines.append("- no execution receipts yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(
        f"- rows={len(rows)} first={row.get('receipt_uuid')} scope={row.get('scope')} "
        f"status={row.get('status')} exit_code={row.get('exit_code')}"
    )
    lines.append(
        f"  command_text={row.get('command_text')} started_at={row.get('started_at')} completed_at={row.get('completed_at')}"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live API test execution receipt packets from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["receipts"], default="receipts")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"api_test_execution_receipts?order=completed_at.desc&limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
