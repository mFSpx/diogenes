#!/usr/bin/env python3
"""Continuous corpus crush loop — hw_gate + River ML governor, no daemon."""
import json, os, subprocess, sys, time, pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INVENTORY = ROOT / "05_OUTPUTS/krampus_inventory/krampus_queue_eligible.jsonl"
CURSOR_FILE = ROOT / "04_RUNTIME/corpus_ingest/cursor.json"


def read_hw() -> dict:
    try:
        from scripts.lucidota_hw_gate import read_hw_state
        return read_hw_state(write_db=True)
    except Exception:
        return {"mode": "saturate", "safe_local_workers": 2}


def read_cursor() -> str:
    try:
        d = json.loads(CURSOR_FILE.read_text())
        return d.get("cursor") or d.get("cursor_after", "")
    except Exception:
        return ""


def recommended_batch(hw: dict) -> int:
    try:
        from scripts.lucidota_river_governor import predict
        return max(4, predict(hw))
    except Exception:
        workers = hw.get("safe_local_workers", 2)
        return max(4, workers * 4)


def report_to_river(hw: dict, bs: int, elapsed: float, ok: bool) -> None:
    try:
        from scripts.lucidota_river_governor import learn
        throughput = bs / elapsed if elapsed > 0 else 0
        learn(hw, batch_size=bs, throughput_per_sec=throughput, oom=not ok)
    except Exception:
        pass


def total_remaining(cursor: str) -> int:
    lines = INVENTORY.read_text().splitlines()
    if not cursor:
        return len(lines)
    for i, line in enumerate(lines):
        try:
            if json.loads(line).get("path", "") == cursor:
                return len(lines) - i - 1
        except Exception:
            pass
    return len(lines)


def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] CORPUS CRUSH START (hw_gate + River)", flush=True)
    runs = 0
    while True:
        hw = read_hw()
        if hw.get("mode") == "throttle" and hw.get("safe_local_workers", 1) <= 1:
            print(f"[HW] throttle {hw.get('reasons')} — sleeping 15s", flush=True)
            time.sleep(15)
            continue

        cursor = read_cursor()
        remaining = total_remaining(cursor)
        if remaining <= 0:
            print(f"[DONE] corpus exhausted after {runs} runs", flush=True)
            break

        bs = recommended_batch(hw)
        cmd = [
            sys.executable, str(ROOT / "scripts/corpus_ingest.py"),
            "--inventory-jsonl", str(INVENTORY),
            "--corpus-map", str(sorted((ROOT / "05_OUTPUTS/goals").glob("corpus_map_*.json"))[-1]),
            "--batch-size", str(bs),
            "--execute",
        ]
        if cursor:
            cmd += ["--start-after", cursor]

        print(f"[RUN {runs+1}] cursor={cursor[:60] if cursor else 'START'} bs={bs} remaining~{remaining}", flush=True)
        t0 = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        elapsed = time.monotonic() - t0
        ok = result.returncode == 0
        if not ok:
            print(f"[ERR] {result.stderr[-200:]}", flush=True)
            time.sleep(5)
        report_to_river(hw, bs, elapsed, ok)
        runs += 1
        time.sleep(0.5)

    print(f"[RECEIPT] {runs} batches completed", flush=True)


if __name__ == "__main__":
    main()
