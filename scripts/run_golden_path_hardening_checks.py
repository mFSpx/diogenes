#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "golden_path_hardening"
MATRIX_FIXTURE = ROOT / "tests/fixtures/matrix_trace/valid_receipt.json"
COMMANDS = [
    [sys.executable, "scripts/canonical_graph_write_scanner.py"],
    [sys.executable, "scripts/spine_authority_checker.py"],
    [sys.executable, "scripts/run_dev_order_methodology_checks.py"],
    [sys.executable, "scripts/golden_path_regression_gate.py", "--dry-run"],
    [sys.executable, "scripts/same_lineage_validator.py", "--receipt", "tests/fixtures/golden_path/valid_receipt_bundle.json"],
    [sys.executable, "scripts/status_ledger_evidence_gate.py", "--audit-ledger"],
    [sys.executable, "-m", "pytest",
        "tests/test_golden_path_regression_gate.py",
        "tests/test_canonical_graph_write_scanner.py",
        "tests/test_spine_authority_checker.py",
        "tests/test_graph_promotion_gate_safety.py",
        "tests/test_same_lineage_validator.py",
        "tests/test_status_ledger_evidence_gate.py",
        "tests/test_proof_kernel.py",
        "tests/test_dev_order_matrix_wrapper.py",
        "tests/test_matrix_trace_checker.py",
        "tests/test_dev_order_gate.py",
        "-q"],
]
FILES_CHANGED = [
    "scripts/golden_path_regression_gate.py",
    "scripts/canonical_graph_write_scanner.py",
    "scripts/spine_authority_checker.py",
    "scripts/same_lineage_validator.py",
    "scripts/status_ledger_evidence_gate.py",
    "scripts/proof_kernel.py",
    "scripts/dev_order_matrix_wrapper.py",
    "scripts/matrix_trace_checker.py",
    "scripts/dev_order_gate.py",
    "scripts/run_dev_order_methodology_checks.py",
    "scripts/run_golden_path_hardening_checks.py",
    "06_SCHEMA/dev_order_matrix_policy.v1.json",
    "06_SCHEMA/proof_object.schema.json",
    "tests/test_proof_kernel.py",
    "tests/test_dev_order_matrix_wrapper.py",
    "tests/test_matrix_trace_checker.py",
    "tests/test_dev_order_gate.py",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    refs = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.startswith("REPORT_PATH="):
            refs.append(line.split("=", 1)[1].strip())
    return {"command": " ".join(cmd), "rc": proc.returncode, "result": "PASS" if proc.returncode == 0 else "FAIL", "report_paths": refs, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}


def load_matrix_trace() -> list[dict[str, Any]]:
    data = json.loads(MATRIX_FIXTURE.read_text(encoding="utf-8"))
    return data["matrix_trace"]


def write_payload(payload: dict[str, Any], out: Path) -> None:
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def main() -> int:
    results = [run(cmd) for cmd in COMMANDS]
    blockers = [r["command"] for r in results if r["rc"] != 0]
    receipts_written = sorted({p for r in results for p in r.get("report_paths", [])})
    payload = {
        "schema": "lucidota.golden_path_hardening.v2",
        "generated_at": now(),
        "matrix_trace": load_matrix_trace(),
        "commands_run": results,
        "commands": results,
        "files_changed": FILES_CHANGED,
        "receipts_written": receipts_written,
        "receipts_created": receipts_written,
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS" if not blockers else "PARTIAL_FAIL",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"golden_path_hardening_{stamp()}.json"
    write_payload(payload, out)

    if payload["verdict"] == "PASS":
        gate = run([sys.executable, "scripts/dev_order_gate.py", "--receipt", rel(out)])
        payload["commands_run"].append(gate)
        payload["commands"].append(gate)
        payload["self_gate"] = gate
        payload["receipts_written"] = sorted(set(payload["receipts_written"] + gate.get("report_paths", [])))
        payload["receipts_created"] = payload["receipts_written"]
        if gate["rc"] != 0:
            payload["blockers"].append("self_dev_order_gate_failed")
            payload["verdict"] = "PARTIAL_FAIL"
        write_payload(payload, out)

    print("REPORT_PATH=" + rel(out))
    print("GOLDEN_PATH_HARDENING=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
