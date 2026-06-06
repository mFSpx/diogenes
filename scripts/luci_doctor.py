#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import urllib.request
from typing import Any

from provider_secret_quarantine import provider_secret_status_json


DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    import urllib.parse

    qs = urllib.parse.urlencode(query or {}, safe=",.")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def live_check(base_url: str, path: str, query: dict[str, str] | None = None) -> dict[str, Any]:
    import urllib.error
    import urllib.parse

    qs = urllib.parse.urlencode(query or {}, safe=",.")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "[]")
        return {"ok": True, "url": url, "rows": payload if isinstance(payload, list) else []}
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}:{exc}"}


def first_row(check: dict[str, Any]) -> dict[str, Any]:
    rows = check.get("rows")
    return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}


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


def route_refs(packet: dict[str, Any]) -> list[str]:
    route_list = packet.get("route_list")
    if not isinstance(route_list, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for route in route_list:
        if not isinstance(route, dict):
            continue
        route_id = route.get("route_id")
        if route_id is None:
            continue
        route_text = str(route_id)
        if route_text not in seen:
            seen.add(route_text)
            refs.append(route_text)
    return refs


def capability_refs(packet: dict[str, Any]) -> list[str]:
    active = packet.get("active_capabilities")
    if not isinstance(active, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for cap in active:
        if not isinstance(cap, dict):
            continue
        key = cap.get("capability_key")
        if key is None:
            continue
        key_text = str(key)
        if key_text not in seen:
            seen.add(key_text)
            refs.append(key_text)
    return refs


def surface_refs(packet: dict[str, Any]) -> list[str]:
    routes = packet.get("route_list")
    if not isinstance(routes, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for route in routes:
        if not isinstance(route, dict):
            continue
        route_id = route.get("route_id")
        if route_id is None:
            continue
        route_text = str(route_id)
        if route_text not in seen:
            seen.add(route_text)
            refs.append(route_text)
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    checks = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "tools": {
            "psql": shutil.which("psql") is not None,
            "git": shutil.which("git") is not None,
            "luci": os.path.exists("luci"),
        },
        "live": {
            "manual_current": live_check(args.base_url, "manual_current", {"limit": "1"}),
            "root_orchestrator_current": live_check(args.base_url, "root_orchestrator_current", {"limit": "1"}),
            "daemon_status": live_check(args.base_url, "daemon_status", {"limit": "1"}),
            "schema_owner_manifest": live_check(args.base_url, "schema_owner_manifest", {"surface_id": "eq.schema_owner_manifest"}),
            "surface_registry": live_check(args.base_url, "surface_registry", {"surface_id": "eq.manual_current"}),
            "renderer_registry": live_check(args.base_url, "renderer_registry", {"limit": "1"}),
            "percyphon_current": live_check(args.base_url, "percyphon_current", {"limit": "1"}),
            "percyphon_village_matrix": live_check(args.base_url, "percyphon_village_matrix", {"limit": "1"}),
            "controller_grant": live_check(args.base_url, "controller_grant", {"grant_key": "eq.default_local_operator"}),
            "agent_thread_runtime": live_check(args.base_url, "agent_thread_runtime", {"thread_key": "eq.root_operator_thread"}),
            "capability_current": live_check(args.base_url, "capability_current", {"limit": "1"}),
            "elastic_shape_current": live_check(args.base_url, "elastic_shape_current", {"limit": "1"}),
            "elastic_shape_latest": live_check(args.base_url, "elastic_shape_latest", {"limit": "1"}),
            "shape_residuals_current": live_check(args.base_url, "shape_residuals_current", {"limit": "1"}),
            "indy_attention_pressure_current": live_check(args.base_url, "indy_attention_pressure_current", {"limit": "1"}),
        },
    }

    rows = {
        "command_registry": live_check(args.base_url, "command_registry", {"limit": "1"}),
        "surface_registry": live_check(args.base_url, "surface_registry", {"limit": "1"}),
        "schema_owner_manifest": live_check(args.base_url, "schema_owner_manifest", {"limit": "5"}),
        "controller_grant": live_check(args.base_url, "controller_grant", {"limit": "5"}),
        "agent_thread_runtime": live_check(args.base_url, "agent_thread_runtime", {"limit": "5"}),
        "daemon_status": live_check(args.base_url, "daemon_status", {"limit": "1"}),
        "percyphon_current": live_check(args.base_url, "percyphon_current", {"limit": "1"}),
        "percyphon_village_matrix": live_check(args.base_url, "percyphon_village_matrix", {"limit": "1"}),
        "capability_current": live_check(args.base_url, "capability_current", {"limit": "1"}),
        "elastic_shape_current": live_check(args.base_url, "elastic_shape_current", {"limit": "1"}),
        "elastic_shape_latest": live_check(args.base_url, "elastic_shape_latest", {"limit": "1"}),
        "shape_residuals_current": live_check(args.base_url, "shape_residuals_current", {"limit": "1"}),
        "indy_attention_pressure_current": live_check(args.base_url, "indy_attention_pressure_current", {"limit": "1"}),
    }

    root = first_row(checks["live"]["root_orchestrator_current"])
    capability = first_row(checks["live"]["capability_current"])
    daemon = first_row(checks["live"]["daemon_status"])
    manual = first_row(checks["live"]["manual_current"])
    percyphon = first_row(checks["live"]["percyphon_current"])
    pressure = first_row(checks["live"]["indy_attention_pressure_current"])
    orchestration = root.get("orchestration") or capability.get("orchestration") or daemon.get("orchestration") or {}
    shape_refs = [
        name
        for name in (
            "elastic_shape_current",
            "elastic_shape_latest",
            "shape_residuals_current",
            "indy_attention_pressure_current",
        )
        if checks["live"][name].get("ok")
    ]
    percyphon_refs = [
        name
        for name in (
            "percyphon_current",
            "percyphon_village_matrix",
        )
        if checks["live"][name].get("ok")
    ]

    payload = {
        "schema": "lucidota.luci.doctor.v1",
        "ok": all(item.get("ok") for item in checks["live"].values()),
        "checks": checks,
        "surfaces": rows,
        "shape_refs": shape_refs,
        "percyphon_refs": percyphon_refs,
        "next_command_refs": dedupe_refs(
            root.get("next_command_refs") or [],
            capability.get("next_command_refs") or [],
            daemon.get("next_command_refs") or [],
        ),
        "orchestration": orchestration,
        "route_refs": route_refs(root),
        "surface_refs": surface_refs(manual),
        "capability_refs": capability_refs(capability),
        "renderer_refs": [
            "renderer_registry",
            "command_registry",
            "schema_owner_manifest",
        ],
        "percyphon_current": percyphon,
        "controller_grant": first_row(checks["live"]["controller_grant"]),
        "agent_thread_runtime": first_row(checks["live"]["agent_thread_runtime"]),
        "provider_secret_status": provider_secret_status_json()["status"],
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print("LUCI DOCTOR")
        for name, item in checks["live"].items():
            print(f"- {name}: {'ok' if item.get('ok') else 'fail'}")
        if payload["next_command_refs"]:
            print(f"next_command_refs={', '.join(payload['next_command_refs'][:8])}")
        if payload["route_refs"]:
            print(f"route_refs={', '.join(payload['route_refs'][:8])}")
        if payload["surface_refs"]:
            print(f"surface_refs={', '.join(payload['surface_refs'][:8])}")
        if payload["capability_refs"]:
            print(f"capability_refs={', '.join(payload['capability_refs'][:8])}")
        if payload["renderer_refs"]:
            print(f"renderer_refs={', '.join(payload['renderer_refs'][:8])}")
        if payload["percyphon_refs"]:
            print(f"percyphon_refs={', '.join(payload['percyphon_refs'][:8])}")
        if payload["shape_refs"]:
            print(f"shape_refs={', '.join(payload['shape_refs'][:8])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
