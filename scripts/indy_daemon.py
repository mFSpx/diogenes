#!/usr/bin/env python3
"""Indy_READs daemon front door.

This is a tiny wrapper around the live DB queue and the existing Indy
respond-once path. It does not scan BOOKS directly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mamba_db_watch
from scripts import indy_runtime_broker

INDY_READS = ROOT / "scripts" / "indy_reads.py"


def respond_once_command() -> list[str]:
    return [sys.executable, str(INDY_READS), "chat", "--respond-once", "--json"]


def run_once(*, base_url: str = mamba_db_watch.DEFAULT_BASE_URL, limit: int = 25, max_items: int = 12) -> dict[str, Any]:
    registry = indy_runtime_broker.registry_snapshot(base_url=base_url)
    poll = mamba_db_watch.poll_once(base_url=base_url, limit=limit, max_items=max_items)
    result: dict[str, Any] = {"registry_snapshot": registry, "poll": poll, "responded": False, "respond_rc": None, "respond_stdout": ""}
    if int(poll.get("row_count") or 0) > 0:
        proc = subprocess.run(respond_once_command(), cwd=str(ROOT), capture_output=True, text=True)
        result["responded"] = proc.returncode == 0
        result["respond_rc"] = proc.returncode
        result["respond_stdout"] = (proc.stdout or proc.stderr or "")[-4000:]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Indy DB-driven daemon wrapper.")
    ap.add_argument("--base-url", default=mamba_db_watch.DEFAULT_BASE_URL)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-items", type=int, default=12)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    while True:
        payload = run_once(base_url=args.base_url, limit=args.limit, max_items=args.max_items)
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False) if args.json else json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        print(text)
        if args.once or not args.loop:
            return 0 if payload.get("respond_rc") in (None, 0) else int(payload["respond_rc"])
        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
