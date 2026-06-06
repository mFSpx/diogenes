#!/usr/bin/env python3
"""Render live CLI process authority receipts from PostgREST."""
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
    lines = ["CLI PROCESS RECEIPTS"]
    if not rows:
        lines.append("- no CLI authority receipts yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    lines.append(f"- receipt {row.get('receipt_uuid')} :: {row.get('status')} pid={row.get('process_pid')} restarts={row.get('restart_count')}")
    lines.append(f"  auth_prompt_seen={row.get('auth_prompt_seen')} auth_injected={row.get('auth_injected')} exit_code={row.get('exit_code')}")
    lines.append(
        f"  stdout_archive_ref={row.get('stdout_archive_ref') or ''} stderr_archive_ref={row.get('stderr_archive_ref') or ''}"
    )
    lines.append(f"  command={row.get('command_line')}")
    lines.append(f"  receipt_path={row.get('receipt_path')}")
    stdout_tail = row.get("stdout_tail") or ""
    stderr_tail = row.get("stderr_tail") or ""
    if stdout_tail:
        lines.append("  stdout_tail:")
        for line in str(stdout_tail).splitlines()[:12]:
            lines.append(f"    {line}")
    if stderr_tail:
        lines.append("  stderr_tail:")
        for line in str(stderr_tail).splitlines()[:12]:
            lines.append(f"    {line}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live CLI process authority receipts from PostgREST.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(
        args.base_url,
        f"cli_process_receipts?select=*,next_command_refs,orchestration&order=received_at.desc&limit={args.limit}",
    )
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
