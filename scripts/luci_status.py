#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any

from provider_secret_quarantine import provider_secret_status_json


DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    import urllib.parse

    qs = urllib.parse.urlencode(query or {}, safe=",.")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def first_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows and isinstance(rows[0], dict) else {}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    active_goal = fetch_json(args.base_url, "active_goal", {"select": "goal_id,title,status,goal,db_law,next_command_refs,updated_at", "limit": "1"})
    root = fetch_json(args.base_url, "root_orchestrator_current", {"select": "orchestrator_id,title,route_list,next_command_refs,orchestration", "limit": "1"})
    daemon = fetch_json(args.base_url, "daemon_status", {"select": "daemon_name,next_command_refs", "limit": "1"})
    capability = fetch_json(args.base_url, "capability_current", {"select": "active_capabilities,next_command_refs", "limit": "1"})
    controller = fetch_json(args.base_url, "controller_grant", {"select": "grant_key,controller_name,effective_status", "grant_key": "eq.default_local_operator"})
    thread = fetch_json(args.base_url, "agent_thread_runtime", {"select": "thread_key,parent_thread_key,thread_owner,runtime_kind", "thread_key": "eq.root_operator_thread"})
    manual = fetch_json(args.base_url, "manual_current", {"select": "manual_id,title,orchestration", "limit": "1"})
    percyphon_current = fetch_json(args.base_url, "percyphon_current", {"select": "vuuid,name,persona,alias,ternary_state,relevance_confidence_bps,authority,updated_at", "limit": "1"})
    percyphon_matrix = fetch_json(args.base_url, "percyphon_village_matrix", {"select": "vuuid,name,persona,alias,ternary_state,relevance_confidence_bps,authority,updated_at", "limit": "1"})
    elastic_shape = fetch_json(args.base_url, "elastic_shape_current", {"select": "receipt_uuid,artifact_uuid,signature,collision_signature,canon_status,route_context", "limit": "1"})
    pressure = fetch_json(args.base_url, "indy_attention_pressure_current", {"select": "receipt_uuid,artifact_uuid,pressure_score,recommended_action,signature,collision_signature,canon_status,route_context", "limit": "1"})
    residuals = fetch_json(args.base_url, "shape_residuals_current", {"select": "residual_count,latest_residual_at", "limit": "1"})
    if not manual and root:
        manual = root

    root_row = first_row(root)
    daemon_row = first_row(daemon)
    capability_row = first_row(capability)
    manual_row = first_row(manual)

    payload = {
        "schema": "lucidota.luci.status.v1",
        "ok": bool(root and manual and active_goal and percyphon_current),
        "active_goal": active_goal[0] if active_goal else {},
        "manual": manual[0] if manual else {},
        "root_orchestrator": root[0] if root else {},
        "daemon_status": daemon[0] if daemon else {},
        "capability_current": capability[0] if capability else {},
        "orchestration": (root[0].get("orchestration") if root and isinstance(root[0], dict) else {}) or (manual[0].get("orchestration") if manual and isinstance(manual[0], dict) else {}),
        "controller_grant": controller[0] if controller else {},
        "agent_thread_runtime": thread[0] if thread else {},
        "percyphon_current": percyphon_current[0] if percyphon_current else {},
        "percyphon_village_matrix": percyphon_matrix[0] if percyphon_matrix else {},
        "elastic_shape_current": elastic_shape[0] if elastic_shape else {},
        "indy_attention_pressure_current": pressure[0] if pressure else {},
        "shape_residuals_current": residuals[0] if residuals else {},
        "provider_secret_status": provider_secret_status_json()["status"],
        "shape_refs": [
            "elastic_shape_current",
            "indy_attention_pressure_current",
            "shape_residuals_current",
        ],
        "percyphon_refs": [
            "percyphon_current",
            "percyphon_village_matrix",
        ],
        "next_command_refs": dedupe_refs(
            root_row.get("next_command_refs") or [],
            daemon_row.get("next_command_refs") or [],
            capability_row.get("next_command_refs") or [],
        ),
        "route_refs": route_refs(root_row),
        "surface_refs": surface_refs(root_row) or surface_refs(manual_row),
        "capability_refs": capability_refs(capability_row),
        "renderer_refs": [
            "renderer_registry",
            "command_registry",
            "schema_owner_manifest",
        ],
    }

    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print("LUCI STATUS")
        print(f"goal={payload['active_goal'].get('title', '')}")
        print(f"manual_id={payload['manual'].get('manual_id', '')}")
        print(f"root_title={payload['root_orchestrator'].get('title', '')}")
        print(f"daemon_name={payload['daemon_status'].get('daemon_name', '')}")
        if payload["percyphon_current"]:
            print(
                "percyphon="
                + f"{payload['percyphon_current'].get('name', '')} "
                + f"persona={payload['percyphon_current'].get('persona', '')} "
                + f"alias={payload['percyphon_current'].get('alias', '')}"
            )
        if payload["indy_attention_pressure_current"]:
            print(
                "pressure="
                + f"{payload['indy_attention_pressure_current'].get('pressure_score', '')} "
                + f"action={payload['indy_attention_pressure_current'].get('recommended_action', '')}"
            )
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
