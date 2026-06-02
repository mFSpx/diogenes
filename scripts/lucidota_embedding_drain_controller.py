#!/usr/bin/env python3
"""Governed launcher for BGE corpus embedding drain jobs.

This controller emits a small decision object and optionally executes
`scripts/corpus_embed_fill_worker.py` with conservative caps under pressure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKER_CMD = [str(ROOT / ".venv" / "bin" / "python3"), str(ROOT / "scripts" / "corpus_embed_fill_worker.py")]
RECEIPT_PREFIX = "embedding_drain_controller_"
MEMORY_FLOOR_MB = 2048
_FILLED_RE = re.compile(r"\bfilled=(\d+)\b")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_slice_from_rung(
    governor_rung: int,
    requested_max_chunks: int,
    requested_concurrency: int,
    requested_http_batch: int,
    requested_jobs: int,
) -> tuple[int, int, int, int]:
    """Return (safe_jobs, max_chunks, concurrency, http_batch)."""
    if governor_rung >= 3:
        return 1, 24, 1, 8
    if governor_rung >= 2:
        return max(1, min(requested_jobs, 2)), max(64, requested_max_chunks // 4), max(1, requested_concurrency // 2), max(8, requested_http_batch // 2)
    return requested_jobs, requested_max_chunks, requested_concurrency, requested_http_batch


def _worker_filled_count(stdout: str | None) -> int | None:
    if not stdout:
        return None
    match = _FILLED_RE.search(stdout)
    if not match:
        return None
    return int(match.group(1))


def _write_receipt(receipt_root: Path, receipt: dict[str, Any]) -> str:
    path = receipt_root / f"{RECEIPT_PREFIX}{_now_iso().replace(':','')}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_mem_available_mb(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def build_snapshot() -> dict[str, Any]:
    env_available = os.environ.get("LUCIDOTA_AVAILABLE_MB")
    if env_available not in (None, ""):
        return {"memory": {"available_mb": int(env_available), "available_mb_source": "env:LUCIDOTA_AVAILABLE_MB"}}
    return {"memory": {"available_mb": read_mem_available_mb(), "available_mb_source": "proc_meminfo"}}


def run_controller(
    *,
    execute: bool,
    snapshot: dict[str, Any],
    governor_rung: int,
    receipt_root: Path | str,
    requested_jobs: int,
    requested_max_chunks: int = 500,
    requested_concurrency: int = 3,
    requested_http_batch: int = 32,
) -> dict[str, Any]:
    """Run one controller decision cycle and optionally launch worker commands."""
    receipt_root = Path(receipt_root)
    receipt_root.mkdir(parents=True, exist_ok=True)
    memory_mb = int(snapshot.get("memory", {}).get("available_mb", 0))

    reasons: list[str] = []
    admit = True
    if memory_mb < MEMORY_FLOOR_MB:
        reasons.append("mem_available_below_floor")
        admit = False

    safe_jobs, max_chunks, concurrency, http_batch = _safe_slice_from_rung(
        governor_rung,
        requested_max_chunks,
        requested_concurrency,
        requested_http_batch,
        requested_jobs,
    )
    # Hard cap concurrency and batch knobs for low-level safety even after rung control.
    max_chunks = max(1, max_chunks)
    concurrency = max(1, min(concurrency, requested_concurrency))
    http_batch = max(1, min(http_batch, requested_http_batch))

    decision = {
        "admit": admit,
        "rung": governor_rung,
        "reasons": reasons,
        "safe_jobs": safe_jobs,
        "worker_args": {
            "max_chunks": max_chunks,
            "concurrency": concurrency,
            "http_batch": http_batch,
        },
    }

    if not admit:
        receipt = {
            "controller": "lucidota_embedding_drain_controller",
            "status": "SKIPPED",
            "decision": decision,
            "generated_at": _now_iso(),
            "requested": {
                "jobs": requested_jobs,
                "max_chunks": requested_max_chunks,
                "concurrency": requested_concurrency,
                "http_batch": requested_http_batch,
            },
            "snapshot": snapshot,
        }
        receipt_path = _write_receipt(receipt_root, receipt)
        return {"status": "SKIPPED", "decision": decision, "receipt_path": receipt_path}

    commands: list[list[str]] = []
    worker_results: list[dict[str, Any]] = []
    any_progress = False
    if execute and safe_jobs > 0:
        for _ in range(safe_jobs):
            cmd = [
                *WORKER_CMD,
                "--max-chunks",
                str(max_chunks),
                "--concurrency",
                str(concurrency),
                "--http-batch",
                str(http_batch),
            ]
            commands.append(cmd)
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            filled = _worker_filled_count(proc.stdout)
            worker_result = {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "filled": filled,
                "progress": bool(filled and filled > 0),
            }
            worker_results.append(worker_result)
            if worker_result["progress"]:
                any_progress = True

    if execute and safe_jobs > 0 and not any_progress:
        decision["reasons"].append("no_embedding_progress")
        receipt = {
            "controller": "lucidota_embedding_drain_controller",
            "status": "SKIPPED",
            "decision": decision,
            "generated_at": _now_iso(),
            "requested": {
                "jobs": requested_jobs,
                "max_chunks": requested_max_chunks,
                "concurrency": requested_concurrency,
                "http_batch": requested_http_batch,
            },
            "snapshot": snapshot,
            "worker_results": worker_results,
            "commands_run": commands,
        }
        receipt_path = _write_receipt(receipt_root, receipt)
        return {
            "status": "SKIPPED",
            "decision": decision,
            "commands_run": commands,
            "worker_results": worker_results,
            "receipt_path": receipt_path,
        }

    receipt = {
        "controller": "lucidota_embedding_drain_controller",
        "status": "PASSED",
        "decision": decision,
        "generated_at": _now_iso(),
        "requested": {
            "jobs": requested_jobs,
            "max_chunks": requested_max_chunks,
            "concurrency": requested_concurrency,
            "http_batch": requested_http_batch,
        },
        "snapshot": snapshot,
        "worker_results": worker_results,
        "commands_run": commands,
    }
    receipt_path = _write_receipt(receipt_root, receipt)
    return {
        "status": "PASSED",
        "decision": decision,
        "commands_run": commands,
        "worker_results": worker_results,
        "receipt_path": receipt_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--governor-rung", type=int, default=0, dest="governor_rung")
    ap.add_argument("--jobs", type=int, default=1, dest="requested_jobs")
    ap.add_argument("--max-chunks", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--http-batch", type=int, default=32)
    args = ap.parse_args()
    snapshot = build_snapshot()
    out = run_controller(
        execute=args.execute,
        snapshot=snapshot,
        governor_rung=args.governor_rung,
        receipt_root=ROOT / "05_OUTPUTS" / "governor",
        requested_jobs=args.requested_jobs,
        requested_max_chunks=args.max_chunks,
        requested_concurrency=args.concurrency,
        requested_http_batch=args.http_batch,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
