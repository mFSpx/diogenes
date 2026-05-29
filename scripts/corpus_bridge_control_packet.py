#!/usr/bin/env python3
"""Generate a kernel control packet for absurd_corpus_job_bridge.py with exact idem key.

The bridge computes idem as:
  sha256(json.dumps({'queue':q,'lane':l,'source_path':s,'state':st,'max_files':mf},
                    sort_keys=True, separators=(',',':'), default=str))

This script replicates that exactly and generates the matching control packet.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kernel_control_packet import absurd_enqueue_packet


def bridge_idem(queue: str, lane: str, source_path: str, state: str, max_files: int) -> str:
    payload = {"queue": queue, "lane": lane, "source_path": source_path, "state": state, "max_files": max_files}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-path", required=True)
    ap.add_argument("--queue", default="korpus")
    ap.add_argument("--lane", default="manifest_inventory")
    ap.add_argument("--state", default="queued")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--authorized-by", default="operator_cli")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    idem = bridge_idem(a.queue, a.lane, a.source_path, a.state, a.max_files)
    pkt = absurd_enqueue_packet(
        queue_name=a.queue, lane=a.lane, source_path=a.source_path,
        idempotency_key=idem, authorized_by=a.authorized_by,
    )
    out = a.out or f"/tmp/corpus_bridge_cp_{a.queue}_{a.max_files}.json"
    Path(out).write_text(json.dumps(pkt))
    print(f"IDEM={idem}")
    print(f"PACKET={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
