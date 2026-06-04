#!/usr/bin/env python3
"""PostgREST RPC client for bounded prompt packets.

The client never reads the database directly. It posts a narrow request to
`/rpc/cloud_packet` and returns the JSON response. This keeps prompt builders on
the HTTP surface, not on raw table dumps.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

MAX_CHARS_CAP = 12_000
MAX_ITEMS_CAP = 32
DEFAULT_BASE_URL = "http://127.0.0.1:3000"


def _clamp_int(value: int | None, *, minimum: int, maximum: int, default: int) -> int:
    try:
        raw = int(value) if value is not None else default
    except Exception:
        raw = default
    return max(minimum, min(maximum, raw))


def build_cloud_packet_request(
    *,
    work_order_id: str,
    max_chars: int = 8_000,
    max_items: int = 12,
    task_type: str = "",
    target_model: str = "",
    include_raw_bodies: bool = False,
) -> dict[str, Any]:
    return {
        "work_order_id": str(work_order_id),
        "max_chars": _clamp_int(max_chars, minimum=256, maximum=MAX_CHARS_CAP, default=8_000),
        "max_items": _clamp_int(max_items, minimum=1, maximum=MAX_ITEMS_CAP, default=12),
        "task_type": (task_type or "").strip(),
        "target_model": (target_model or "").strip(),
        "include_raw_bodies": bool(include_raw_bodies),
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "lucidota-prompt-api-client/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8") or "null"
    return json.loads(raw)


def cloud_packet(
    *,
    base_url: str = DEFAULT_BASE_URL,
    work_order_id: str,
    max_chars: int = 8_000,
    max_items: int = 12,
    task_type: str = "",
    target_model: str = "",
    include_raw_bodies: bool = False,
    timeout: float = 5.0,
) -> dict[str, Any]:
    request_payload = build_cloud_packet_request(
        work_order_id=work_order_id,
        max_chars=max_chars,
        max_items=max_items,
        task_type=task_type,
        target_model=target_model,
        include_raw_bodies=include_raw_bodies,
    )
    return _post_json(f"{base_url.rstrip('/')}/rpc/cloud_packet", request_payload, timeout=timeout)


@dataclass(frozen=True)
class CloudPacketResult:
    payload: dict[str, Any]
    request: dict[str, Any]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch a bounded prompt packet from PostgREST.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--work-order-id", required=True)
    ap.add_argument("--max-chars", type=int, default=8_000)
    ap.add_argument("--max-items", type=int, default=12)
    ap.add_argument("--task-type", default="")
    ap.add_argument("--target-model", default="")
    ap.add_argument("--include-raw-bodies", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    payload = cloud_packet(
        base_url=args.base_url,
        work_order_id=args.work_order_id,
        max_chars=args.max_chars,
        max_items=args.max_items,
        task_type=args.task_type,
        target_model=args.target_model,
        include_raw_bodies=args.include_raw_bodies,
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
