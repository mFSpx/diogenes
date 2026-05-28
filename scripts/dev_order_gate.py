#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "dev_order_matrix"
CHECKER = ROOT / "scripts" / "matrix_trace_checker.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("json_not_object")
    return data


def parse_report_path(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return line.split("=", 1)[1].strip()
    return None


def truthy_any(receipt: dict[str, Any], keys: list[str]) -> bool:
    return any(receipt.get(key) is True for key in keys)


def nonempty_list(receipt: dict[str, Any], key: str) -> bool:
    value = receipt.get(key)
    return isinstance(value, list) and len(value) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Blocking gate for matrix-wrapped dev order receipts")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--allow-canonical-graph-writes", action="store_true")
    parser.add_argument("--allow-canonical-graph-materialization", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    input_receipt = Path(args.receipt)
    if not input_receipt.is_absolute():
        input_receipt = ROOT / input_receipt
    blockers: list[str] = []
    commands_run: list[dict[str, Any]] = []
    checker_report_path: str | None = None
    checker_report: dict[str, Any] | None = None
    receipt: dict[str, Any] = {}

    try:
        receipt = load_json(input_receipt)
    except Exception as exc:
        blockers.append(f"input_receipt_unreadable:{exc}")

    cmd = [sys.executable, str(CHECKER), "--receipt", rel(input_receipt)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    checker_report_path = parse_report_path(proc.stdout)
    commands_run.append({
        "command": " ".join(cmd),
        "rc": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "report_path": checker_report_path,
    })
    if checker_report_path:
        try:
            checker_report = load_json(ROOT / checker_report_path)
        except Exception as exc:
            blockers.append(f"checker_report_unreadable:{exc}")
    else:
        blockers.append("matrix_checker_report_missing")
    if proc.returncode != 0:
        blockers.append("matrix_trace_checker_failed")

    final_verdict = receipt.get("verdict")
    if final_verdict == "PASS" and proc.returncode != 0:
        blockers.append("final_pass_without_matrix_trace_pass")
    if not args.allow_canonical_graph_writes and truthy_any(receipt, ["canonical_graph_writes", "canonical_graph_writes_performed"]):
        blockers.append("unexpected_canonical_graph_writes")
    if not args.allow_canonical_graph_materialization and truthy_any(receipt, ["canonical_graph_materialization", "canonical_graph_materialization_performed", "materialization_performed"]):
        blockers.append("unexpected_canonical_graph_materialization")
    if final_verdict == "PASS":
        if not nonempty_list(receipt, "commands_run"):
            blockers.append("pass_requires_commands_run")
        if not nonempty_list(receipt, "files_changed"):
            blockers.append("pass_requires_files_changed")
        if not nonempty_list(receipt, "receipts_written"):
            blockers.append("pass_requires_receipts_written")

    payload = {
        "schema": "lucidota.dev_order_matrix.gate_receipt.v1",
        "generated_at": utc_now(),
        "input_receipt": rel(input_receipt),
        "matrix_checker_rc": proc.returncode,
        "matrix_checker_report": checker_report_path,
        "matrix_checker_verdict": checker_report.get("verdict") if checker_report else None,
        "commands_run": commands_run,
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "verdict": "PASS" if not blockers else "FAIL",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT_DIR / f"dev_order_gate_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("DEV_ORDER_GATE=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
