#!/usr/bin/env python3
"""Tiny ternary Command Envelope Router scaffold for FairyFuse/LUCIDOTA.

No external network calls. No hard-coded DB user. Hardware/BitNet execution is STUB
until a real FairyFuse/BitNet backend is wired and benchmarked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
TERNARY_DIMS = 12
try:
    from services.fairyfuse.fairyfuse_backend import route_command as fairyfuse_route_command
except Exception:  # pragma: no cover - import path/environment dependent
    fairyfuse_route_command = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def payload_hash(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> str:
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_primitive_id(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> int:
    digest = payload_hash(raw_command, normalized_intent, context)
    return int(digest[:8], 16) % 25


def ternary_vector(raw_command: str, normalized_intent: str, context: dict[str, Any], dims: int = TERNARY_DIMS) -> list[int]:
    digest = hashlib.sha256((raw_command + "\0" + normalized_intent + "\0" + json.dumps(context, sort_keys=True)).encode()).digest()
    values = []
    for idx in range(dims):
        values.append((digest[idx] % 3) - 1)
    return values


def confidence_bps(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> int:
    base = 4500
    if normalized_intent.strip():
        base += 1800
    if raw_command.strip():
        base += 1200
    if context:
        base += 800
    return max(0, min(9900, base))


def parse_context(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value


def optional_db_schema_check(database_url: str | None, lens_key: str | None) -> dict[str, Any]:
    if not database_url:
        return {"checked": False, "reason": "DATABASE_URL not provided; dry stub path"}
    try:
        import psycopg  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"checked": False, "error": f"psycopg unavailable: {exc}"}
    try:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('lucidota_ternary.lens_registry')")
                reg = cur.fetchone()[0]
                if reg is None:
                    return {"checked": True, "ok": False, "error": "missing lucidota_ternary.lens_registry"}
                if lens_key:
                    cur.execute("SELECT classification::text, fast_path_compatible FROM lucidota_ternary.lens_registry WHERE lens_key=%s", (lens_key,))
                    row = cur.fetchone()
                    if row is None:
                        return {"checked": True, "ok": False, "error": f"missing lens_key={lens_key}"}
                    return {"checked": True, "ok": True, "lens_key": lens_key, "classification": row[0], "fast_path_compatible": bool(row[1])}
                return {"checked": True, "ok": True, "lens_key": None}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"checked": True, "ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="LUCIDOTA tiny ternary Command Envelope Router scaffold")
    parser.add_argument("--dry-run", action="store_true", help="do not write DB rows or call runtime backend")
    parser.add_argument("--raw-command", required=True)
    parser.add_argument("--normalized-intent", required=True)
    parser.add_argument("--context", default="{}", help="JSON object with surface/workflow context")
    parser.add_argument("--lens-key", default="command_envelope_router_stub")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()

    context = parse_context(args.context)
    sha = payload_hash(args.raw_command, args.normalized_intent, context)
    db_check = optional_db_schema_check(args.database_url, args.lens_key) if not args.dry_run else {"checked": False, "reason": "dry_run"}

    if fairyfuse_route_command is not None and args.lens_key == "command_envelope_router_v0":
        routed = fairyfuse_route_command(args.raw_command, args.normalized_intent, context).to_dict()
        output = {
            "schema": "lucidota.ternary_router.v1",
            "status": routed["status"],
            "dry_run": args.dry_run,
            "payload_sha256": routed["payload_sha256"],
            "primitive_id": routed["primitive_id"],
            "ternary_vector": routed["ternary_vector"],
            "confidence_bps": routed["confidence_bps"],
            "backend": {
                "fairyfuse": routed["backend"],
                "bitnet_path": "NOT_USED_BY_SYMBOLIC_V0",
                "fast_path_claim": routed["fast_path_preserved"],
                "fast_path_contract": routed["fast_path_contract"],
                "reason": "FairyFuse v0 symbolic backend wired; this is not a general ternary LoRA or BitNet adapter."
            },
            "db_check": db_check,
            "created_at": routed["created_at"],
        }
    else:
        output = {
            "schema": "lucidota.ternary_router.v0.stub",
            "status": "STUB_BACKEND_NOT_WIRED",
            "dry_run": args.dry_run,
            "payload_sha256": sha,
            "primitive_id": stable_primitive_id(args.raw_command, args.normalized_intent, context),
            "ternary_vector": ternary_vector(args.raw_command, args.normalized_intent, context),
            "confidence_bps": confidence_bps(args.raw_command, args.normalized_intent, context),
            "backend": {
                "fairyfuse": "STUB",
                "bitnet_path": "STUB",
                "fast_path_claim": False,
                "reason": "No real low-bit backend wired or benchmarked for selected lens."
            },
            "db_check": db_check,
            "created_at": utc_now(),
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    if db_check.get("checked") and not db_check.get("ok"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
