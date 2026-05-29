#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "dev_order_matrix"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def run(cmd: list[str], expected: set[int] | None = None) -> dict[str, Any]:
    expected = expected or {0}
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    report_paths: list[str] = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.startswith("REPORT_PATH="):
            report_paths.append(line.split("=", 1)[1].strip())
    return {
        "command": " ".join(cmd),
        "rc": proc.returncode,
        "expected_rc": sorted(expected),
        "result": "PASS" if proc.returncode in expected else "FAIL",
        "report_paths": report_paths,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
    }


def load_valid_matrix_trace() -> list[dict[str, Any]]:
    path = ROOT / "tests/fixtures/matrix_trace/valid_receipt.json"
    return json.loads(path.read_text(encoding="utf-8"))["matrix_trace"]


def main() -> int:
    commands = [
        ([sys.executable, "-m", "py_compile", "scripts/dev_order_matrix_wrapper.py", "scripts/matrix_trace_checker.py", "scripts/dev_order_gate.py", "scripts/run_dev_order_methodology_checks.py"], {0}),
        ([sys.executable, "scripts/dev_order_matrix_wrapper.py", "--input", "tests/fixtures/dev_orders/simple_dev_order.txt", "--output", "04_RUNTIME/dev_orders/wrapped/simple_dev_order.wrapped.txt", "--receipt-out", "05_OUTPUTS/dev_order_matrix/wrap_manual_test.json"], {0}),
        ([sys.executable, "scripts/matrix_trace_checker.py", "--receipt", "tests/fixtures/matrix_trace/valid_receipt.json"], {0}),
        ([sys.executable, "scripts/matrix_trace_checker.py", "--receipt", "tests/fixtures/matrix_trace/missing_constraint_receipt.json"], {4}),
        ([sys.executable, "scripts/matrix_trace_checker.py", "--receipt", "tests/fixtures/matrix_trace/wrong_order_receipt.json"], {4}),
        ([sys.executable, "scripts/matrix_trace_checker.py", "--receipt", "tests/fixtures/matrix_trace/prose_only_matrix_receipt.json"], {4}),
        ([sys.executable, "scripts/matrix_trace_checker.py", "--receipt", "tests/fixtures/matrix_trace/pass_with_incomplete_matrix_receipt.json"], {4}),
        ([sys.executable, "scripts/dev_order_gate.py", "--receipt", "tests/fixtures/matrix_trace/valid_receipt.json"], {0}),
        ([sys.executable, "-m", "pytest", "tests/test_dev_order_matrix_wrapper.py", "tests/test_matrix_trace_checker.py", "tests/test_dev_order_gate.py", "-q"], {0}),
    ]
    results = [run(cmd, expected) for cmd, expected in commands]
    blockers = [r["command"] for r in results if r["result"] != "PASS"]
    payload = {
        "schema": "lucidota.dev_order_matrix.methodology_receipt.v1",
        "generated_at": utc_now(),
        "policy_path": "06_SCHEMA/dev_order_matrix_policy.v1.json",
        "matrix_trace": load_valid_matrix_trace(),
        "commands_run": results,
        "files_changed": [
            "06_SCHEMA/dev_order_matrix_policy.v1.json",
            "scripts/dev_order_matrix_wrapper.py",
            "scripts/matrix_trace_checker.py",
            "scripts/dev_order_gate.py",
            "scripts/run_dev_order_methodology_checks.py",
            "tests/test_dev_order_matrix_wrapper.py",
            "tests/test_matrix_trace_checker.py",
            "tests/test_dev_order_gate.py",
        ],
        "receipts_written": sorted({p for r in results for p in r.get("report_paths", [])}),
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "canonical_graph_writes_performed": False,
        "blockers": blockers,
        "verdict": "PASS" if not blockers else "FAIL",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"dev_order_methodology_{stamp()}.json"
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    if payload["verdict"] == "PASS":
        gate = run([sys.executable, "scripts/dev_order_gate.py", "--receipt", rel(out)], {0})
        payload["commands_run"].append(gate)
        payload["receipts_written"] = sorted(set(payload["receipts_written"] + gate.get("report_paths", [])))
        payload["self_gate"] = gate
        if gate["result"] != "PASS":
            payload["blockers"].append("self_dev_order_gate_failed")
            payload["verdict"] = "FAIL"
        out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("DEV_ORDER_METHODOLOGY=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
