from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "tests/fixtures/matrix_trace/valid_receipt.json"


def run_gate(path: Path, expect: int):
    p = subprocess.run([sys.executable, "scripts/dev_order_gate.py", "--receipt", str(path)], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == expect, p.stdout + p.stderr
    return p


def report(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text())
    raise AssertionError(stdout)


def mutate(tmp_path: Path, mutator) -> Path:
    data = json.loads(VALID.read_text())
    mutator(data)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(data))
    return path


def test_gate_accepts_valid_receipt():
    r = report(run_gate(VALID, 0).stdout)
    assert r["verdict"] == "PASS"


def test_gate_rejects_missing_matrix_trace(tmp_path):
    path = mutate(tmp_path, lambda d: d.pop("matrix_trace", None))
    r = report(run_gate(path, 4).stdout)
    assert "matrix_trace_checker_failed" in r["blockers"]


def test_gate_rejects_matrix_checker_failure(tmp_path):
    def m(d): d["matrix_trace"][0]["constraint_order"] = list(reversed(d["matrix_trace"][0]["constraint_order"]))
    r = report(run_gate(mutate(tmp_path, m), 4).stdout)
    assert "matrix_trace_checker_failed" in r["blockers"]


def test_gate_rejects_pass_without_files_changed(tmp_path):
    path = mutate(tmp_path, lambda d: d.__setitem__("files_changed", []))
    r = report(run_gate(path, 4).stdout)
    assert "pass_requires_files_changed" in r["blockers"]


def test_gate_rejects_pass_without_commands_run(tmp_path):
    path = mutate(tmp_path, lambda d: d.__setitem__("commands_run", []))
    r = report(run_gate(path, 4).stdout)
    assert "pass_requires_commands_run" in r["blockers"]


def test_gate_rejects_pass_without_receipts_written(tmp_path):
    path = mutate(tmp_path, lambda d: d.__setitem__("receipts_written", []))
    r = report(run_gate(path, 4).stdout)
    assert "pass_requires_receipts_written" in r["blockers"]


def test_gate_rejects_unexpected_canonical_graph_writes(tmp_path):
    path = mutate(tmp_path, lambda d: d.__setitem__("canonical_graph_writes", True))
    r = report(run_gate(path, 4).stdout)
    assert "unexpected_canonical_graph_writes" in r["blockers"]


def test_gate_rejects_unexpected_canonical_graph_materialization(tmp_path):
    path = mutate(tmp_path, lambda d: d.__setitem__("canonical_graph_materialization", True))
    r = report(run_gate(path, 4).stdout)
    assert "unexpected_canonical_graph_materialization" in r["blockers"]
