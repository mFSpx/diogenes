#!/usr/bin/env python3
"""API-first Indy runtime broker.

This stays on the PostgREST surface: it snapshots registry routes and fetches
bounded prompt packets via `/rpc/cloud_packet`. It does not read raw tables or
scan repo payloads directly.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prompt_api_client

DEFAULT_BASE_URL = os.environ.get("POSTGREST_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
REGISTRY_ROUTES = (
    "api_workflow_registry",
    "active_goal",
    "capability_registry",
    "canon_current",
    "flow_receipts",
    "flow_specs",
    "manual_current",
    "model_routing_current",
    "model_registry",
    "provider_registry",
    "workflow_registry",
)

LOCAL_MODEL_ROLES = ("router", "classifier", "summarizer", "embedder", "reranker", "thinker", "watcher", "treelite_gate")


def fetch_json(path: str, query: dict[str, str] | None = None, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> tuple[int, Any, str]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8") or "null"
            return resp.status, json.loads(raw), ""
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        return exc.code, None, body
    except Exception as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def registry_snapshot(*, base_url: str = DEFAULT_BASE_URL, routes: tuple[str, ...] = REGISTRY_ROUTES) -> dict[str, Any]:
    status, openapi, error = fetch_json("", None, base_url=base_url)
    openapi_paths = openapi.get("paths", {}) if isinstance(openapi, dict) else {}
    model_status, model_rows, model_error = fetch_json("model_registry", {"limit": "50"}, base_url=base_url)
    provider_status, provider_rows, provider_error = fetch_json("provider_registry", {"limit": "50"}, base_url=base_url)
    local_model_roles = {
        role: next(
            (
                {
                    "model_id": row.get("model_id"),
                    "slot_name": row.get("slot_name"),
                    "role": row.get("role"),
                    "loadout_id": row.get("loadout_id"),
                    "expected_vram_mb": row.get("expected_vram_mb"),
                    "benchmark_status": row.get("benchmark_status"),
                    "notes": row.get("notes"),
                }
                for row in model_rows
                if isinstance(row, dict) and row.get("role") == role
            ),
            None,
        )
        for role in LOCAL_MODEL_ROLES
    } if isinstance(model_rows, list) else {role: None for role in LOCAL_MODEL_ROLES}
    route_rows: list[dict[str, Any]] = []
    for route in routes:
        route_status, body, route_error = fetch_json(route, {"limit": "3"}, base_url=base_url)
        rows = body if isinstance(body, list) else []
        route_rows.append(
            {
                "route": f"/{route}",
                "openapi_methods": sorted(openapi_paths.get(f"/{route}", {}).keys()),
                "http_status": route_status,
                "sample_rows": len(rows),
                "fields": sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else [],
                "error": route_error,
            }
        )
    return {
        "schema": "lucidota.indy_runtime_broker.snapshot.v1",
        "postgrest_base_url": base_url,
        "openapi_status": status,
        "openapi_error": error,
        "model_registry_status": model_status,
        "model_registry_error": model_error,
        "provider_registry_status": provider_status,
        "provider_registry_error": provider_error,
        "local_model_roles": local_model_roles,
        "route_rows": route_rows,
    }


def choose_local_model(*, role: str, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any] | None:
    """Return the live local model row for a role, or None if the registry has no exact match."""
    status, body, _ = fetch_json("model_registry", {"limit": "50"}, base_url=base_url)
    if status != 200 or not isinstance(body, list):
        return None
    for row in body:
        if isinstance(row, dict) and row.get("role") == role and row.get("active", True):
            return row
    return None


def build_cloud_packet(
    *,
    work_order_id: str,
    base_url: str = DEFAULT_BASE_URL,
    max_chars: int = 8000,
    max_items: int = 12,
    task_type: str = "",
    target_model: str = "",
    include_raw_bodies: bool = False,
) -> dict[str, Any]:
    return prompt_api_client.cloud_packet(
        base_url=base_url,
        work_order_id=work_order_id,
        max_chars=max_chars,
        max_items=max_items,
        task_type=task_type,
        target_model=target_model,
        include_raw_bodies=include_raw_bodies,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Indy runtime broker over PostgREST.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_snapshot = sub.add_parser("snapshot", help="Fetch safe registry route snapshot.")
    p_snapshot.add_argument("--json", action="store_true")

    p_packet = sub.add_parser("packet", help="Fetch bounded prompt packet from PostgREST.")
    p_packet.add_argument("--work-order-id", required=True)
    p_packet.add_argument("--max-chars", type=int, default=8000)
    p_packet.add_argument("--max-items", type=int, default=12)
    p_packet.add_argument("--task-type", default="")
    p_packet.add_argument("--target-model", default="")
    p_packet.add_argument("--include-raw-bodies", action="store_true")
    p_packet.add_argument("--json", action="store_true")

    args = ap.parse_args()
    if args.cmd == "snapshot":
        payload = registry_snapshot(base_url=args.base_url)
    else:
        payload = build_cloud_packet(
            base_url=args.base_url,
            work_order_id=args.work_order_id,
            max_chars=args.max_chars,
            max_items=args.max_items,
            task_type=args.task_type,
            target_model=args.target_model,
            include_raw_bodies=args.include_raw_bodies,
        )

    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
