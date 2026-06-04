#!/usr/bin/env python3
"""Render live prompt-ledger packets from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"

ROUTE_FOR_MODE = {
    "recent": "prompt_recent",
    "filed": "prompts_filed",
    "links": "prompt_work_order_links",
    "unlinked": "prompt_unlinked",
    "catalog": "prompt_catalog_status",
}


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(mode: str, rows: list[dict[str, Any]], *, raw: bool = False) -> str:
    title = f"PROMPT {'RAW ' if raw else ''}{mode.upper()}".strip()
    lines = [title]
    if not rows:
        lines.append("- no prompt ledger rows yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    if mode == "catalog":
        lines.append(
            f"- prompt_count={row.get('prompt_count')} filed_count={row.get('filed_count')} "
            f"linked_count={row.get('linked_count')} unlinked_count={row.get('unlinked_count')}"
        )
        lines.append(
            f"  decomposed_count={row.get('decomposed_count')} executed_count={row.get('executed_count')} "
            f"latest_received_at={row.get('latest_received_at')}"
        )
    elif mode == "links":
        lines.append(f"- rows={len(rows)} first_prompt_id={row.get('prompt_id')} work_order_uuid={row.get('work_order_uuid')}")
    else:
        lines.append(
            f"- rows={len(rows)} first_prompt_id={row.get('prompt_id')} "
            f"status={row.get('status')} linked_goal_id={row.get('linked_goal_id')}"
        )
        if row.get("raw_prompt_text"):
            lines.append(f"  raw_prompt_text={str(row.get('raw_prompt_text'))[:140]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live prompt ledger packets from PostgREST.")
    ap.add_argument("mode", choices=["raw", *sorted(ROUTE_FOR_MODE.keys())])
    ap.add_argument("raw_mode", nargs="?", choices=sorted(ROUTE_FOR_MODE.keys()))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    raw = args.mode == "raw"
    mode = args.raw_mode if raw else args.mode
    if raw and not mode:
        ap.error("raw requires recent|filed|links|unlinked|catalog")
    route = ROUTE_FOR_MODE[mode]
    rows = fetch_json(args.base_url, f"{route}?order=received_at.desc&limit={args.limit}" if mode != "catalog" else f"{route}?limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(mode, rows, raw=raw), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
