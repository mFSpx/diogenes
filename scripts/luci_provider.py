#!/usr/bin/env python3
"""Render live provider registry current packet from PostgREST."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from provider_secret_quarantine import provider_secret_status_json

DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def fetch_json(base_url: str, path: str, query: dict[str, str] | None = None) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(query or {}, safe=",.()")
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["PROVIDER CURRENT"]
    if not rows:
        lines.append("- no provider registry packet yet")
        return "\n".join(lines) + "\n"
    row = rows[0]
    row["secret_status"] = provider_secret_status_json()["status"]
    if "controller_grant" not in row:
        grant_rows = fetch_json(DEFAULT_BASE_URL, "controller_grant", {"grant_key": "eq.default_local_operator"})
        if grant_rows:
            row["controller_grant"] = grant_rows[0]
    if "agent_thread_runtime" not in row:
        thread_rows = fetch_json(DEFAULT_BASE_URL, "agent_thread_runtime", {"thread_key": "eq.root_operator_thread"})
        if thread_rows:
            row["agent_thread_runtime"] = thread_rows[0]
    summary = row.get("provider_summary") or {}
    lines.append(
        f"- packet {row.get('provider_packet_id')} :: providers={summary.get('provider_count')} "
        f"active={summary.get('active_count')} kinds={summary.get('kind_count')} local={summary.get('local_count')} cloud={summary.get('cloud_count')}"
    )
    lines.append(
        f"  active_rows={len(row.get('active_providers') or [])} "
        f"kind_keys={sorted((row.get('kind_breakdown') or {}).keys())}"
    )
    kind_names = summary.get("provider_kind_names") or []
    if kind_names:
        lines.append(f"  provider_kind_names={', '.join(str(name) for name in kind_names[:12])}")
    routing_notes = row.get("routing_notes") or {}
    if routing_notes:
        lines.append("  routing_notes:")
        for key, value in routing_notes.items():
            lines.append(f"    {key}: {value}")
    secret_status = row.get("secret_status") or {}
    if secret_status:
        lines.append(f"  secret_status={json.dumps(secret_status, sort_keys=True)}")
    controller_grant = row.get("controller_grant") or {}
    if controller_grant:
        lines.append(
            "  controller_grant="
            + json.dumps(
                {
                    "grant_key": controller_grant.get("grant_key"),
                    "status": controller_grant.get("effective_status") or controller_grant.get("status"),
                    "max_parallel_threads": controller_grant.get("max_parallel_threads"),
                    "max_spend": controller_grant.get("max_spend"),
                    "receipt_uuid": controller_grant.get("receipt_uuid"),
                },
                sort_keys=True,
            )
        )
    agent_thread_runtime = row.get("agent_thread_runtime") or {}
    if agent_thread_runtime:
        lines.append(
            "  agent_thread_runtime="
            + json.dumps(
                {
                    "thread_key": agent_thread_runtime.get("thread_key"),
                    "runtime_kind": agent_thread_runtime.get("runtime_kind"),
                    "thread_owner": agent_thread_runtime.get("thread_owner"),
                    "budget_scope": agent_thread_runtime.get("budget_scope"),
                    "receipt_uuid": agent_thread_runtime.get("receipt_uuid"),
                },
                sort_keys=True,
            )
        )
    active_rows = row.get("active_providers") or []
    if active_rows:
        lines.append("  active_providers:")
        for provider in active_rows[:12]:
            lines.append(
                f"    - {provider.get('provider_key')} :: {provider.get('provider_kind')} "
                f"[{provider.get('active')}] -> {provider.get('default_model')}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Show live provider registry packet from PostgREST.")
    ap.add_argument("mode", nargs="?", choices=["current"], default="current")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    rows = fetch_json(args.base_url, f"provider_current?order=refreshed_at.desc&limit={args.limit}")
    if rows:
        rows[0]["secret_status"] = provider_secret_status_json()["status"]
        grant_rows = fetch_json(args.base_url, "controller_grant", {"grant_key": "eq.default_local_operator"})
        if grant_rows:
            rows[0]["controller_grant"] = grant_rows[0]
        thread_rows = fetch_json(args.base_url, "agent_thread_runtime", {"thread_key": "eq.root_operator_thread"})
        if thread_rows:
            rows[0]["agent_thread_runtime"] = thread_rows[0]
    if args.json:
        print(json.dumps(rows, sort_keys=True, ensure_ascii=False))
    else:
        print(render(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
