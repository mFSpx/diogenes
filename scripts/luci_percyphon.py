#!/usr/bin/env python3
"""Render Percyphon village current/matrix runtime surfaces from PostgREST."""
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
    with urllib.request.urlopen(url, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["PERCYPHON VILLAGE"]
    if not rows:
        lines.append("- no Percyphon rows yet")
        return "\n".join(lines) + "\n"
    for row in rows[:5]:
        lines.append(
            f"- vuuid={row.get('vuuid')} name={row.get('name')} persona={row.get('persona')} alias={row.get('alias')} ternary={row.get('ternary_state')} confidence_bps={row.get('relevance_confidence_bps')}"
        )
        if row.get("packet"):
            pkt = row["packet"]
            lines.append(
                f"  packet slot_count={pkt.get('slot_count')} identity={pkt.get('identity_slot_count')} procedural={pkt.get('procedural_slot_count')} authority={pkt.get('authority')}"
            )
        slot_1 = row.get("slot_001")
        slot_28 = row.get("slot_028")
        slot_29 = row.get("slot_029")
        slot_128 = row.get("slot_128")
        if slot_1 or slot_28 or slot_29 or slot_128:
            lines.append(
                "  slots="
                + ", ".join(
                    part for part in [
                        f"slot_001={slot_1}",
                        f"slot_028={slot_28}",
                        f"slot_029={slot_29}",
                        f"slot_128={slot_128}",
                    ] if part.split("=", 1)[1] != "None"
                )
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show runtime Percyphon village matrix/current packets from PostgREST.")
    ap.add_argument("surface", nargs="?", default="current", choices=("current", "matrix"))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    surface_map = {
        "current": "percyphon_current",
        "matrix": "percyphon_village_matrix",
    }
    rows = fetch_json(args.base_url, surface_map[args.surface], {"limit": str(args.limit)})
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
