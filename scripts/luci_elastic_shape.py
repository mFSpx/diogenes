#!/usr/bin/env python3
"""Render runtime elastic shape receipts and residuals from PostgREST."""
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
    lines = ["ELASTIC SHAPE ROUTER"]
    if not rows:
        lines.append("- no runtime shape rows yet")
        return "\n".join(lines) + "\n"
    for row in rows:
        artifact = row.get("artifact_uuid")
        signature = row.get("signature") or row.get("collision_signature")
        dims = row.get("dimensions")
        fidelity = row.get("fidelity")
        collision = row.get("collision")
        lines.append(f"- artifact={artifact} sig={signature} dims={dims} fidelity={fidelity} collision={collision}")
        if row.get("pressure_score") is not None:
            lines.append(f"  pressure_score={row['pressure_score']} recommended_action={row.get('recommended_action', '')}")
        if row.get("active_resonances"):
            top = row["active_resonances"][:4]
            lines.append("  resonances=" + ", ".join(f"{name}:{score}" for name, score in top))
        refs = row.get("next_command_refs") or []
        if refs:
            lines.append(f"  next_command_refs={', '.join(str(ref) for ref in refs)}")
        orchestration = row.get("orchestration") or {}
        if orchestration:
            lines.append(
                "  orchestration="
                + ", ".join(f"{k}={orchestration.get(k)}" for k in ("mode", "sub_orchestrator_priority") if orchestration.get(k) is not None)
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show runtime elastic shape packets from PostgREST.")
    ap.add_argument("surface", nargs="?", default="current", choices=("current", "latest", "residuals", "pressure"))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    surface_map = {
        "current": "elastic_shape_current",
        "latest": "elastic_shape_latest",
        "residuals": "shape_residuals_current",
        "pressure": "indy_attention_pressure_current",
    }
    rows = fetch_json(args.base_url, f"{surface_map[args.surface]}?limit={args.limit}")
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
