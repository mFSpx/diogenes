#!/usr/bin/env python3
"""Always-on FairyFuse ternary router.

CPU-bound resident route for LUCIDOTA dual-engine inference.  It keeps the
FairyFuse backend initialized, memory-maps packed ternary weights when present,
and exposes status/route/daemon commands for Bytewax and Absurd workers.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.fairyfuse.fairyfuse_backend import resident_engine_from_env, route_command  # noqa: E402

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value


def route_packet(packet: dict[str, Any]) -> dict[str, Any]:
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or "bytewax_rete_bandit")
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }
    route = route_command(text[:4096], intent, context).to_dict()
    route["engine_channel"] = "cpu_fairyfuse_ternary"
    route["outbound_state"] = "draft_only"
    return route


def emit_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


def status_cmd(args: argparse.Namespace) -> int:
    emit_json(resident_engine_from_env().status())
    return 0


def route_cmd(args: argparse.Namespace) -> int:
    route = route_command(args.raw_command, args.normalized_intent, parse_context(args.context)).to_dict()
    route["engine_channel"] = "cpu_fairyfuse_ternary"
    route["outbound_state"] = "draft_only"
    emit_json(route)
    return 0


def daemon_cmd(args: argparse.Namespace) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    heartbeat_path = Path(args.heartbeat)
    if not heartbeat_path.is_absolute():
        heartbeat_path = ROOT / heartbeat_path
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    engine = resident_engine_from_env()
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    cycles = 0
    while not stop["flag"]:
        cycles += 1
        status = engine.status()
        receipt = {
            "schema": "lucidota.fairyfuse.ternary_router_heartbeat.v1",
            "created_at": now_z(),
            "cycle": cycles,
            "pid": os.getpid(),
            "always_on": True,
            "engine_channel": "cpu_fairyfuse_ternary",
            "status": status,
        }
        with heartbeat_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        if args.max_cycles and cycles >= args.max_cycles:
            break
        time.sleep(max(0.1, float(args.idle_sleep)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LUCIDOTA FairyFuse resident ternary router")
    sub = parser.add_subparsers(dest="cmd", required=False)
    p_status = sub.add_parser("status")
    p_status.set_defaults(func=status_cmd)
    p_route = sub.add_parser("route")
    p_route.add_argument("--raw-command", default="")
    p_route.add_argument("--normalized-intent", default="bytewax_rete_bandit")
    p_route.add_argument("--context", default="{}")
    p_route.set_defaults(func=route_cmd)
    p_daemon = sub.add_parser("daemon")
    p_daemon.add_argument("--idle-sleep", type=float, default=float(os.environ.get("LUCIDOTA_FAIRYFUSE_IDLE_SLEEP", "2.0")))
    p_daemon.add_argument("--max-cycles", type=int, default=0)
    p_daemon.add_argument("--heartbeat", default=str(DEFAULT_HEARTBEAT.relative_to(ROOT)))
    p_daemon.set_defaults(func=daemon_cmd)
    parser.set_defaults(func=status_cmd)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
