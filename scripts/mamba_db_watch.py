#!/usr/bin/env python3
"""DB-first Mamba watch helper for Indy_READs.

This polls PostgREST queue/responses views and emits compact queue summaries.
No BOOKS filesystem scan, no raw-corpus crawl, no direct DB connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.environ.get("POSTGREST_BASE_URL", "http://127.0.0.1:3000").rstrip("/")


def fetch_rows(route: str, *, base_url: str = DEFAULT_BASE_URL, limit: int = 25) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode({"limit": str(limit)})
    url = f"{base_url.rstrip('/')}/{route.lstrip('/')}?{qs}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "[]")
    return payload if isinstance(payload, list) else []


def _stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def compact_queue_rows(rows: list[dict[str, Any]], *, max_items: int = 12, window_kind: str = "tumbling") -> dict[str, Any]:
    capped = rows[: max(1, min(int(max_items), 32))]
    event_ids = [str(row.get("event_id") or row.get("id") or "") for row in capped if row.get("event_id") or row.get("id")]
    source_hashes = [
        _stable_hash(
            str(row.get("event_id") or row.get("id") or ""),
            str(row.get("clean_text") or row.get("raw_text") or ""),
            str(row.get("sender_id") or ""),
        )
        for row in capped
    ]
    receipt_refs = [str(row.get("receipt_id") or "") for row in capped if row.get("receipt_id")]
    rooms = sorted({str(row.get("room_id") or "") for row in capped if row.get("room_id")})
    senders = sorted({str(row.get("sender_id") or "") for row in capped if row.get("sender_id")})
    statuses = sorted({str(row.get("processed_status") or "") for row in capped if row.get("processed_status")})
    needs_cloud_reasoning = any(bool(str(row.get("raw_text") or row.get("clean_text") or "").strip()) for row in capped)
    summary_text = f"queue={len(rows)} capped={len(capped)} rooms={len(rooms)} senders={len(senders)}"
    return {
        "schema": "lucidota.mamba_db_watch.compact_queue.v1",
        "source": "lucidota_canon.indy_queue",
        "topic": "indy",
        "object_type": "dialogue",
        "window_kind": window_kind,
        "event_count": len(capped),
        "dropped_raw_bodies": len(capped),
        "summary": {"text": summary_text, "rooms": rooms, "senders": senders, "statuses": statuses},
        "features": {"queue_count": len(rows), "rooms": len(rooms), "senders": len(senders)},
        "scores": {"local_score": min(1.0, len(capped) / 10.0), "treelite_score": 0.0, "treelite_lane": "slow"},
        "needs_cloud_reasoning": needs_cloud_reasoning,
        "event_ids": event_ids,
        "source_hashes": source_hashes,
        "receipt_refs": receipt_refs,
        "timestamps": {
            "received_at": [row.get("received_at") for row in capped],
            "created_at": [row.get("created_at") for row in capped],
            "updated_at": [row.get("updated_at") for row in capped],
        },
    }


def poll_once(*, base_url: str = DEFAULT_BASE_URL, limit: int = 25, max_items: int = 12) -> dict[str, Any]:
    rows = fetch_rows("indy_queue", base_url=base_url, limit=limit)
    compact = compact_queue_rows(rows, max_items=max_items)
    compact["row_count"] = len(rows)
    compact["visible_route"] = "/indy_queue"
    return compact


def main() -> int:
    ap = argparse.ArgumentParser(description="Poll PostgREST Indy queue and emit compact windows.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-items", type=int, default=12)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    while True:
        payload = poll_once(base_url=args.base_url, limit=args.limit, max_items=args.max_items)
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False) if args.json else json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        if not args.loop:
            return 0
        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
