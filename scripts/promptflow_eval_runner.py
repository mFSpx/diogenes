#!/usr/bin/env python3
"""Run a PromptFlow batch/eval and write a LUCIDOTA receipt.

PromptFlow is a validation lane, not authority. This wrapper keeps the run
visible even when PromptFlow/Dolt/Postgres are unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DSN = os.getenv("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")
DEFAULT_OUTPUT_DIR = ROOT / "05_OUTPUTS" / "promptflow_traces"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "receipts"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_pf() -> str | None:
    local = ROOT / ".venv" / "bin" / "pf"
    if local.exists():
        return str(local)
    return shutil.which("pf")


def promptflow_run_outputs_path(run_id: str, *, home: Path | None = None) -> Path:
    root = home or Path.home()
    return root / ".promptflow" / ".runs" / run_id / "outputs.jsonl"


def _flatten(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def row_passes(row: dict[str, Any]) -> bool:
    for value in _flatten(row):
        if isinstance(value, str) and value.strip().lower() == "pass":
            return True
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.7:
            return True
    return False


def summarize_outputs(path: Path) -> dict[str, Any]:
    total = 0
    passed = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row_passes(row):
                passed += 1
    return {"total_rows": total, "pass_count": passed, "pass_rate": (passed / total if total else 0.0)}


def write_receipt(payload: dict[str, Any], run_id: str) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPT_DIR / f"promptflow_eval_{run_id}_{now_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def insert_evolution_gate(run_id: str, flow: str, stats: dict[str, Any]) -> str:
    try:
        import psycopg2
    except Exception as exc:
        return f"postgres_unavailable:{type(exc).__name__}"
    try:
        conn = psycopg2.connect(STATE_DSN)
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_learning.evolution_gate_run (fire, reason, batch_stats)
                VALUES (%s, %s, %s::jsonb)
                """,
                (
                    bool(stats["pass_rate"] > 0.75),
                    f"promptflow_eval run_id={run_id} pass_rate={stats['pass_rate']}",
                    json.dumps({"run_id": run_id, "flow": flow, **stats}, sort_keys=True),
                ),
            )
        conn.close()
        return "postgres_insert_ok"
    except Exception as exc:
        return f"postgres_insert_skipped:{type(exc).__name__}:{str(exc)[:180]}"


def run_promptflow(flow: Path, data: Path | None, run_id: str, output_dir: Path) -> dict[str, Any]:
    pf = find_pf()
    if not pf:
        return {"available": False, "exit_code": 0, "stdout": "", "stderr": "PromptFlow not installed"}
    try:
        cmd = [pf, "run", "create", "--flow", str(flow), "--name", run_id, "--stream"]
        if data:
            cmd.extend(["--data", str(data)])
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    except FileNotFoundError:
        return {"available": False, "exit_code": 1, "stdout": "", "stderr": "PromptFlow executable disappeared"}
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        return {"available": True, "exit_code": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{run_id}_stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (output_dir / f"{run_id}_stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")

    try:
        show = subprocess.run([pf, "run", "show-details", "--name", run_id, "--all-results"], cwd=ROOT, text=True, capture_output=True)
        (output_dir / f"{run_id}_show_details.txt").write_text(show.stdout + show.stderr, encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        (output_dir / f"{run_id}_show_details.txt").write_text(f"show-details unavailable: {type(exc).__name__}: {exc}", encoding="utf-8")

    run_outputs = promptflow_run_outputs_path(run_id)
    if run_outputs.exists():
        shutil.copyfile(run_outputs, output_dir / "outputs.jsonl")
        shutil.copyfile(run_outputs, output_dir / f"{run_id}_outputs.jsonl")
    return {"available": True, "exit_code": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run PromptFlow eval and write receipt")
    ap.add_argument("--flow", required=True)
    ap.add_argument("--data")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    flow = Path(args.flow)
    data = Path(args.data) if args.data else None
    out = Path(args.output_dir)
    if not flow.is_absolute():
        flow = ROOT / flow
    if data and not data.is_absolute():
        data = ROOT / data
    if not out.is_absolute():
        out = ROOT / out

    pf_result = run_promptflow(flow, data, args.run_id, out)
    stats = summarize_outputs(out / "outputs.jsonl")
    pg_status = insert_evolution_gate(args.run_id, str(flow), stats)
    receipt = {
        "schema": "lucidota.promptflow_eval_runner.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": args.run_id,
        "flow": str(flow),
        "data": str(data) if data else None,
        "runtime_status": "alive",
        "eval_quality": "unproven",
        "promptflow": pf_result,
        "stats": stats,
        "postgres_status": pg_status,
        "dolt_status": "not_attempted_by_laptop_wrapper",
    }
    receipt_path = write_receipt(receipt, args.run_id)
    if args.json:
        print(json.dumps(receipt, sort_keys=True, default=str))
    else:
        print(
            f"run_id={args.run_id} runtime=alive eval_quality=unproven pass_rate={stats['pass_rate']:.4f} receipt={receipt_path}"
        )
    # Sidecar-only by design: PromptFlow is optional operator/operator-call proof
    # surface and should never hard-fail local local ops by exit status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
