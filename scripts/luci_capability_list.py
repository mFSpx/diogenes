#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    import urllib.parse

    qs = urllib.parse.urlencode(query or {}, safe=",.")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def dedupe_refs(*groups: list[Any]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for ref in group:
            ref_text = str(ref)
            if ref_text and ref_text not in seen:
                seen.add(ref_text)
                refs.append(ref_text)
    return refs


def capability_refs(rows: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("capability_key")
        if key is None:
            continue
        key_text = str(key)
        if key_text not in seen:
            seen.add(key_text)
            refs.append(key_text)
    return refs


def surface_refs(summary_row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for ref in summary_row.get("next_command_refs") or []:
        ref_text = str(ref)
        if ref_text and ref_text not in refs:
            refs.append(ref_text)
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = fetch_json(args.base_url, "capability_registry", {"order": "capability_key.asc", "limit": "200"})
    summary = fetch_json(args.base_url, "capability_current", {"limit": "1"})
    summary_row = summary[0] if summary and isinstance(summary[0], dict) else {}
    orchestration = summary_row.get("orchestration") if isinstance(summary_row.get("orchestration"), dict) else {}
    next_command_refs = dedupe_refs(
        summary_row.get("next_command_refs") or [],
        summary_row.get("capability_refs") or [],
        summary_row.get("surface_refs") or [],
        summary_row.get("renderer_refs") or [],
        ["capability_current", "capability_registry"],
    )
    payload = {
        "schema": "lucidota.luci.capability_list.v1",
        "ok": bool(rows),
        "capabilities": rows,
        "current": summary_row,
        "orchestration": orchestration,
        "next_command_refs": next_command_refs,
        "surface_refs": surface_refs(summary_row),
        "capability_refs": capability_refs(rows),
        "renderer_refs": [
            "renderer_registry",
            "command_registry",
            "schema_owner_manifest",
        ],
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print("LUCI CAPABILITY LIST")
        for row in rows[:30]:
            print(f"- {row.get('capability_key', '')}: {row.get('capability_name', '')}")
        if orchestration:
            priority = orchestration.get("sub_orchestrator_priority") or []
            print(f"orchestration_mode={orchestration.get('mode', '')}")
            if priority:
                print(f"orchestration_priority={', '.join(map(str, priority[:8]))}")
        if payload["next_command_refs"]:
            print(f"next_command_refs={', '.join(payload['next_command_refs'][:8])}")
        if payload["surface_refs"]:
            print(f"surface_refs={', '.join(payload['surface_refs'][:8])}")
        if payload["capability_refs"]:
            print(f"capability_refs={', '.join(payload['capability_refs'][:8])}")
        if payload["renderer_refs"]:
            print(f"renderer_refs={', '.join(payload['renderer_refs'][:8])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
