#!/usr/bin/env python3
"""Render live capability registry rows from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show the live capability registry rows from PostgREST.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    route_url = f"{args.base_url.rstrip('/')}/capability_registry?order=updated_at.desc&limit={args.limit}"
    try:
        with urllib.request.urlopen(route_url, timeout=5) as resp:
            rows = json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        if args.json:
            print(json.dumps({"ok": False, "status": "ROUTE_NOT_LIVE", "source_url": route_url, "error": f"HTTPError:{exc.code}:{body}"}, sort_keys=True, ensure_ascii=False))
        else:
            print("CAPABILITY_REGISTRY_ROUTE_NOT_LIVE")
            print(f"ROUTE={route_url}")
            print(f"ERROR=HTTPError:{exc.code}:{body}")
        raise SystemExit(2)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "status": "ERROR", "source_url": route_url, "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True, ensure_ascii=False))
        else:
            print("CAPABILITY_REGISTRY_ERROR")
            print(f"ROUTE={route_url}")
            print(f"ERROR={type(exc).__name__}:{exc}")
        raise SystemExit(2)

    result = {"ok": True, "status": "FOUND", "source_url": route_url, "rows": rows, "payload": rows}
    if args.json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    else:
        print("ROUTE=/capability_registry")
        print(f"URL={route_url}")
        print(f"ROW_COUNT={len(rows) if isinstance(rows, list) else 0}")
        first = rows[0] if isinstance(rows, list) and rows else {}
        if isinstance(first, dict) and first:
            print(f"CAPABILITY_KEY={first.get('capability_key') or ''}")
            print(f"CAPABILITY_NAME={first.get('capability_name') or ''}")
            print(f"ACTIVE={first.get('active') if 'active' in first else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
