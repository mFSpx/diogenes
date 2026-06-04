#!/usr/bin/env python3
"""Budgeted LUCI command speed probe."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from statistics import median


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    ap = argparse.ArgumentParser(prog="luci-speed-probe")
    ap.add_argument("--command", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--p95-budget-ms", type=float, default=2500.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    runs = max(1, min(args.runs, 20))
    durations: list[float] = []
    samples = []
    for i in range(runs):
        started = time.perf_counter()
        proc = subprocess.run(
            shlex.split(args.command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        durations.append(elapsed_ms)
        samples.append({
            "run": i + 1,
            "duration_ms": round(elapsed_ms, 3),
            "returncode": proc.returncode,
            "stdout_bytes": len(proc.stdout.encode()),
            "stderr_bytes": len(proc.stderr.encode()),
        })
        if proc.returncode != 0:
            payload = {
                "schema": "lucidota.luci_speed_probe.v1",
                "status": "FAIL",
                "command": args.command,
                "runs": i + 1,
                "error": "command_failed",
                "returncode": proc.returncode,
                "stderr_preview": proc.stderr[:500],
                "samples": samples,
            }
            print(json.dumps(payload, sort_keys=True) if args.json else payload)
            return 1
    p95 = percentile(durations, 95)
    payload = {
        "schema": "lucidota.luci_speed_probe.v1",
        "status": "PASS" if p95 <= args.p95_budget_ms else "FAIL",
        "command": args.command,
        "runs": runs,
        "median_ms": round(median(durations), 3),
        "p95_ms": round(p95, 3),
        "p95_budget_ms": args.p95_budget_ms,
        "samples": samples,
    }
    print(json.dumps(payload, sort_keys=True) if args.json else payload)
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
