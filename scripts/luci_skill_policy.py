#!/usr/bin/env python3
"""Render live skill policy packet from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["SKILL POLICY CURRENT"]
    if not rows:
        lines.append("- no skill policy packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(f"- {row.get('policy_id')} :: {row.get('policy_title')} [{row.get('status')}]")
    lines.append(f"  source_ref={row.get('source_ref')}")
    lines.append("  policy_text:")
    for line in str(row.get("policy_text") or "").splitlines()[:16]:
        lines.append(f"    {line}")
    detail = row.get("detail") or {}
    if detail:
        lines.append("  detail:")
        for key, value in detail.items():
            lines.append(f"    {key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live skill policy packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = fetch_json(args.base_url, "skill_policy_current?limit=5")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
