#!/usr/bin/env python3
"""Launch Groq work-order batches with async fanout and receipts."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def plan_launches(batches: list[list[dict[str, Any]]], max_cloud_workers: int = 24) -> dict[str, Any]:
    selected_workers = min(max(1, len(batches)), max(1, int(max_cloud_workers)))
    return {
        "launch_count": len(batches),
        "selected_workers": selected_workers,
        "batches": batches,
    }


async def _launch_one_batch(batch: list[dict[str, Any]], *, execute: bool, model: str, max_tokens: int, timeout_sec: float) -> dict[str, Any]:
    task = json.dumps(batch, sort_keys=True, separators=(",", ":"))
    cmd = [
        sys.executable,
        "scripts/groq_goal_delegate.py",
        "--kind",
        "code-slice",
        "--task",
        task,
        "--model",
        model,
        "--max-tokens",
        str(max_tokens),
        "--json",
    ]
    if execute:
        cmd.append("--execute")
    started = time.perf_counter()
    env = dict(os.environ)
    env.setdefault("GROQ_LOAD_DOTENV", "1")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=ROOT,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        stdout = stdout_b.decode("utf-8", "replace")
        stderr = stderr_b.decode("utf-8", "replace")
    except asyncio.TimeoutError:
        proc.kill()
        stdout = ""
        stderr = "timeout"
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    report_path = next((line.split("=", 1)[1].strip() for line in stdout.splitlines() if line.startswith("REPORT_PATH=")), "")
    blockers: list[str] = []
    subreceipt = {}
    if report_path:
        try:
            subreceipt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
            blockers.extend(subreceipt.get("blockers") or [])
        except Exception:
            blockers.append("subreceipt_parse_failed")
    return {
        "returncode": proc.returncode if proc.returncode is not None else 124,
        "elapsed_ms": elapsed_ms,
        "report_path": report_path,
        "subreceipt": subreceipt,
        "blockers": blockers,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "batch_size": len(batch),
    }


async def launch_batches(
    batches: list[list[dict[str, Any]]],
    *,
    execute: bool = False,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 256,
    timeout_sec: float = 120.0,
    max_cloud_workers: int = 24,
) -> dict[str, Any]:
    plan = plan_launches(batches, max_cloud_workers=max_cloud_workers)
    semaphore = asyncio.Semaphore(plan["selected_workers"])
    launched: list[dict[str, Any]] = []

    async def run(batch: list[dict[str, Any]]) -> dict[str, Any]:
        async with semaphore:
            return await _launch_one_batch(batch, execute=execute, model=model, max_tokens=max_tokens, timeout_sec=timeout_sec)

    launched = await asyncio.gather(*(run(batch) for batch in batches))
    return {
        "schema": "lucidota.groq.batch_launcher.v1",
        "generated_at": now(),
        "launch_count": len(batches),
        "selected_workers": plan["selected_workers"],
        "execute_performed": bool(execute),
        "model": model,
        "max_tokens": max_tokens,
        "timeout_sec": timeout_sec,
        "launches": launched,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-cloud-workers", type=int, default=24)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--model", default=os.environ.get("GROQ_GOAL_MODEL", "llama-3.1-8b-instant"))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import groq_workorder_compiler as gwc

    source = Path(args.source) if args.source else gwc.latest_gap_workflow_path()
    compiled = gwc.compile_workorders(source, batch_size=args.batch_size)
    report = asyncio.run(
        launch_batches(
            compiled["batches"],
            execute=args.execute,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            max_cloud_workers=args.max_cloud_workers,
        )
    )
    report["compiled_source_path"] = compiled["source_path"]
    report["compiled_workflow_count"] = compiled["workflow_count"]
    report["compiled_batch_size"] = compiled["batch_size"]
    report["compiled_batch_count"] = compiled["batch_count"]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"groq_batch_launcher_{stamp()}.json"
    report["report_path"] = rel(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    print("GROQ_BATCH_LAUNCHER=PASS" if all((item.get("returncode") == 0) for item in report["launches"]) else "GROQ_BATCH_LAUNCHER=BLOCKED")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    return 0 if all((item.get("returncode") == 0) for item in report["launches"]) else 4


if __name__ == "__main__":
    raise SystemExit(main())
