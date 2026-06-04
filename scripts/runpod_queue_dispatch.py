#!/usr/bin/env python3
"""RunPod queue dispatcher for Talkie training receipts.

Reads ``04_RUNTIME/RUNPOD_ACCEL/TALKIE_TRAINING_QUEUE.jsonl`` by default,
resolves per-job receipt paths robustly, skips any job whose receipt already
exists, and can optionally execute pending commands. The script stays local-
only: no model blob uploads and no Bonsai lane changes.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_PATH = Path("04_RUNTIME/RUNPOD_ACCEL/TALKIE_TRAINING_QUEUE.jsonl")
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS" / "runpod" / "runpod_queue_dispatch_receipt.json"
RECEIPT_KEYS = ("receipt_path", "receipt", "receipt_file", "receipt_json", "output_receipt", "expected_receipt")
COMMAND_KEYS = ("command", "dispatch_command", "launch_command", "remote_command")


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def resolve_input_path(raw: Path) -> Path:
    if raw.is_absolute():
        return raw
    candidate_roots = (Path.cwd(), ROOT)
    for base in candidate_roots:
        candidate = base / raw
        if candidate.exists():
            return candidate
    if raw == Path("receipts/TALKIE_TRAINING_QUEUE.jsonl"):
        for candidate in (
            ROOT / "04_RUNTIME" / "RUNPOD_ACCEL" / "TALKIE_TRAINING_QUEUE.jsonl",
            Path.cwd() / "04_RUNTIME" / "RUNPOD_ACCEL" / "TALKIE_TRAINING_QUEUE.jsonl",
        ):
            if candidate.exists():
                return candidate
    return Path.cwd() / raw


def _as_path(value: Any) -> Path | None:
    if not value:
        return None
    if isinstance(value, dict):
        for key in ("path", "receipt_path", "file", "json", "value"):
            nested = value.get(key)
            if nested:
                return Path(str(nested))
        return None
    return Path(str(value))


def resolve_receipt_path(raw: Any, queue_path: Path) -> Path:
    path = _as_path(raw)
    if path is None:
        raise ValueError("missing receipt path")
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, queue_path.parent / path, ROOT / path):
        if candidate.exists():
            return candidate
    if queue_path.parent.name == "receipts":
        return queue_path.parent / path
    return Path.cwd() / path


def load_queue(queue_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(queue_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{queue_path}:{line_no}: invalid JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{queue_path}:{line_no}: queue row must be a JSON object")
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _command_list(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(part) for part in raw]
    if isinstance(raw, str):
        return shlex.split(raw)
    return None


def build_report(queue_path: Path, *, execute: bool = False) -> dict[str, Any]:
    rows = load_queue(queue_path)
    jobs: list[dict[str, Any]] = []
    executed = skipped = blocked = failed = 0

    for row in rows:
        receipt_raw = _first_present(row, RECEIPT_KEYS)
        job_name = str(row.get("label") or row.get("name") or row.get("job") or row.get("target_file") or f"line-{row['_line_no']}")
        report: dict[str, Any] = {
            "line_no": row["_line_no"],
            "label": job_name,
            "receipt_path": None,
            "receipt_exists": False,
            "state": "pending",
        }

        try:
            receipt_path = resolve_receipt_path(receipt_raw, queue_path)
            report["receipt_path"] = rel(receipt_path)
            report["receipt_exists"] = receipt_path.exists()
        except Exception as exc:
            report["state"] = "blocked_missing_receipt_path"
            report["error"] = str(exc)
            blocked += 1
            jobs.append(report)
            continue

        if report["receipt_exists"]:
            report["state"] = "skipped_existing_receipt"
            skipped += 1
            jobs.append(report)
            continue

        command = _command_list(_first_present(row, COMMAND_KEYS))
        if not command:
            report["state"] = "blocked_missing_command"
            report["error"] = "queue row has no command to dispatch"
            blocked += 1
            jobs.append(report)
            continue

        report["command"] = command
        if not execute:
            report["state"] = "pending_no_execute"
            jobs.append(report)
            continue

        cp = subprocess.run(command, text=True, capture_output=True, check=False, cwd=ROOT)
        report.update(
            {
                "state": "executed" if cp.returncode == 0 else "failed",
                "returncode": cp.returncode,
                "stdout_tail": cp.stdout[-4000:],
                "stderr_tail": cp.stderr[-4000:],
            }
        )
        if cp.returncode == 0:
            executed += 1
        else:
            failed += 1
        jobs.append(report)

    if failed or blocked:
        status = "PARTIAL"
    elif execute:
        status = "NOOP" if executed == 0 and skipped == 0 else "PASS"
    else:
        status = "DRY_RUN"
    return {
        "schema": "lucidota.runpod.queue_dispatch.v1",
        "generated_at": now_z(),
        "status": status,
        "queue_path": rel(queue_path),
        "queue_exists": queue_path.exists(),
        "queue_rows": len(rows),
        "jobs": jobs,
        "skipped_count": skipped,
        "executed_count": executed,
        "blocked_count": blocked,
        "failed_count": failed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit or dispatch the Talkie RunPod queue.")
    parser.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH), help="Path to TALKIE_TRAINING_QUEUE.jsonl")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT, help="Where to write the local dispatcher receipt")
    parser.add_argument("--execute", action="store_true", help="Run pending queue commands")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout")
    args = parser.parse_args(argv)

    queue_path = resolve_input_path(Path(args.queue_path))
    if not queue_path.exists():
        report = {
            "schema": "lucidota.runpod.queue_dispatch.v1",
            "generated_at": now_z(),
            "status": "BLOCKED",
            "queue_path": rel(queue_path),
            "queue_exists": False,
            "queue_rows": 0,
            "jobs": [],
            "skipped_count": 0,
            "executed_count": 0,
            "blocked_count": 1,
            "failed_count": 0,
            "error": f"missing queue file: {queue_path}",
        }
    else:
        report = build_report(queue_path, execute=args.execute)

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    report["receipt_path"] = rel(args.receipt)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RECEIPT_PATH=" + rel(args.receipt))
    print("RUNPOD_QUEUE_DISPATCH=" + report["status"])
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "NOOP", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
