#!/usr/bin/env python3
"""Render or request a bounded cloud packet via PostgREST RPC."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:3000"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import prompt_api_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Request a bounded cloud packet from PostgREST RPC.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--work-order-id", required=True)
    ap.add_argument("--max-chars", type=int, default=8_000)
    ap.add_argument("--max-items", type=int, default=12)
    ap.add_argument("--task-type", default="")
    ap.add_argument("--target-model", default="")
    ap.add_argument("--include-raw-bodies", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    payload: dict[str, Any] = prompt_api_client.cloud_packet(
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
