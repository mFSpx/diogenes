#!/usr/bin/env python3
"""Render live direct bible subtree packets from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show the live direct bible subtree packet from PostgREST.")
    ap.add_argument("--root-id", required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    route_url = f"{args.base_url.rstrip('/')}/api_bible_subtree?root_id=eq.{urllib.parse.quote(args.root_id)}&limit=1"
    try:
        with urllib.request.urlopen(route_url, timeout=5) as resp:
            rows = json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        if args.json:
            print(json.dumps({"ok": False, "status": "ROUTE_NOT_LIVE", "source_url": route_url, "error": f"HTTPError:{exc.code}:{body}"}, sort_keys=True, ensure_ascii=False))
        else:
            print("API_BIBLE_SUBTREE_ROUTE_NOT_LIVE")
            print(f"ROUTE={route_url}")
            print(f"ERROR=HTTPError:{exc.code}:{body}")
        raise SystemExit(2)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "status": "ERROR", "source_url": route_url, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True, ensure_ascii=False))
        else:
            print("API_BIBLE_SUBTREE_ERROR")
            print(f"ROUTE={route_url}")
            print(f"ERROR={type(exc).__name__}:{exc}")
        raise SystemExit(2)
    result = {"ok": True, "status": "FOUND", "source_url": route_url, "rows": rows, "payload": rows}
    if args.json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    else:
        first = rows[0] if isinstance(rows, list) and rows else {}
        print("ROUTE=/api_bible_subtree")
        print(f"URL={route_url}")
        print(f"ROW_COUNT={len(rows) if isinstance(rows, list) else 0}")
        print(f"ROOT_ID={first.get('root_id') if isinstance(first, dict) else ''}")
        print(f"TITLE={first.get('title') if isinstance(first, dict) else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
