#!/usr/bin/env python3
"""CKDOG1 control-packet helpers for Python workers.

This mirrors lucidota-core ControlPacket hashing so ABSURD/Postgres worker scripts can require
kernel-style authorization receipts before enqueueing executable work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DOMAIN_PREFIX = b"lucidota-domain-hash-v1\0"
CONTROL_PACKET_NAMESPACE = "ckdog1-control-packet"
VALID_ACTIONS = {"add_mandatory", "add_optional", "disable_lane"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def domain_hash_hex(namespace: str, value: Any) -> str:
    if not namespace.strip():
        raise ValueError("domain hash namespace must be non-empty")
    h = hashlib.sha256()
    h.update(DOMAIN_PREFIX)
    h.update(namespace.encode("utf-8"))
    h.update(b"\0")
    h.update(canonical_json_bytes(value))
    return h.hexdigest()


def control_packet_hash_payload(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane": packet.get("lane", ""),
        "action": packet.get("action", ""),
        "authorized_by": packet.get("authorized_by", ""),
        "prev_packet_hash": packet.get("prev_packet_hash"),
        "payload": packet.get("payload", {}),
    }


def make_control_packet(
    *,
    lane: str,
    action: str,
    authorized_by: str,
    payload: dict[str, Any] | None = None,
    prev_packet_hash: str | None = None,
) -> dict[str, Any]:
    if not lane.strip():
        raise ValueError("control packet missing lane")
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid control packet action: {action}")
    if not authorized_by.strip():
        raise ValueError("control packet missing authorized_by")
    packet = {
        "lane": lane,
        "action": action,
        "authorized_by": authorized_by,
        "prev_packet_hash": prev_packet_hash,
        "packet_hash": "",
        "payload": payload or {},
    }
    packet["packet_hash"] = domain_hash_hex(CONTROL_PACKET_NAMESPACE, control_packet_hash_payload(packet))
    return packet


def verify_control_packet(packet: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        if not str(packet.get("authorized_by", "")).strip():
            return False, "control packet missing authorized_by"
        if packet.get("action") not in VALID_ACTIONS:
            return False, f"invalid control packet action: {packet.get('action')}"
        expected = domain_hash_hex(CONTROL_PACKET_NAMESPACE, control_packet_hash_payload(packet))
        if expected != packet.get("packet_hash"):
            return False, f"control packet hash mismatch expected={expected} got={packet.get('packet_hash')}"
        if packet_is_expired(packet):
            return False, "control packet expired"
        return True, None
    except Exception as exc:
        return False, repr(exc)


def require_control_packet(packet: dict[str, Any]) -> None:
    ok, error = verify_control_packet(packet)
    if not ok:
        raise ValueError(error or "invalid control packet")


def absurd_enqueue_packet(
    *,
    queue_name: str,
    lane: str,
    source_path: str,
    idempotency_key: str,
    authorized_by: str,
    expires_at: str | None = None,
) -> dict[str, Any]:
    payload = {
        "capability": "absurd_enqueue",
        "queue_name": queue_name,
        "lane": lane,
        "source_path": source_path,
        "idempotency_key": idempotency_key,
        "canonical_mutation_allowed": False,
    }
    if expires_at is not None:
        payload["expires_at"] = expires_at
    return make_control_packet(
        lane=f"absurd:{queue_name}:{lane}",
        action="add_mandatory",
        authorized_by=authorized_by,
        payload=payload,
    )


def packet_is_expired(packet: dict[str, Any], *, now: datetime | None = None) -> bool:
    expires_at = packet.get("payload", {}).get("expires_at")
    if not expires_at:
        return False
    text = str(expires_at).replace("Z", "+00:00")
    try:
        expiry = datetime.fromisoformat(text)
    except ValueError:
        return True
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) >= expiry


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--lane", required=True)
    mk.add_argument("--action", choices=sorted(VALID_ACTIONS), default="add_mandatory")
    mk.add_argument("--authorized-by", required=True)
    mk.add_argument("--payload-json", default="{}")
    mk.add_argument("--prev-packet-hash")
    vf = sub.add_parser("verify")
    vf.add_argument("--packet-json")
    vf.add_argument("--packet-file")
    args = ap.parse_args()
    if args.cmd == "make":
        packet = make_control_packet(
            lane=args.lane,
            action=args.action,
            authorized_by=args.authorized_by,
            payload=json.loads(args.payload_json),
            prev_packet_hash=args.prev_packet_hash,
        )
        print(json.dumps(packet, sort_keys=True, separators=(",", ":")))
        return 0
    text = args.packet_json if args.packet_json is not None else Path(args.packet_file).read_text(encoding="utf-8")
    packet = json.loads(text)
    ok, error = verify_control_packet(packet)
    print(json.dumps({"authorized": ok, "packet_hash": packet.get("packet_hash"), "error": error}, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
