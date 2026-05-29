from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "tests/fixtures/matrix_trace/valid_receipt.json"


def run_checker(path: Path, expect: int):
    p = subprocess.run([sys.executable, "scripts/matrix_trace_checker.py", "--receipt", str(path)], cwd=ROOT, text=True, capture_output=True)
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
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(data))
    return p


def test_valid_receipt_passes():
    r = report(run_checker(VALID, 0).stdout)
    assert r["verdict"] == "PASS"
    assert r["final_pass_allowed"] is True


def test_missing_task_fails(tmp_path):
    path = mutate(tmp_path, lambda d: d["matrix_trace"].pop())
    r = report(run_checker(path, 4).stdout)
    assert "T7" in r["missing_tasks"]


def test_wrong_order_fails():
    r = report(run_checker(ROOT / "tests/fixtures/matrix_trace/wrong_order_receipt.json", 4).stdout)
    assert r["bad_orders"]


def test_missing_constraint_fails():
    r = report(run_checker(ROOT / "tests/fixtures/matrix_trace/missing_constraint_receipt.json", 4).stdout)
    assert r["missing_constraints"]


def test_duplicate_constraint_fails(tmp_path):
    def m(d):
        d["matrix_trace"][0]["constraint_trace"][1] = copy.deepcopy(d["matrix_trace"][0]["constraint_trace"][0])
    r = report(run_checker(mutate(tmp_path, m), 4).stdout)
    assert r["duplicate_constraints"]


def test_unknown_constraint_fails(tmp_path):
    def m(d):
        d["matrix_trace"][0]["constraint_trace"][0]["constraint"] = "Z"
    r = report(run_checker(mutate(tmp_path, m), 4).stdout)
    assert r["unknown_constraints"]


def test_applied_constraint_without_evidence_fails(tmp_path):
    def m(d):
        row = d["matrix_trace"][0]["constraint_trace"][0]
        for field in ["files_changed", "tests", "commands", "receipt_fields", "failure_conditions"]:
            row[field] = []
    r = report(run_checker(mutate(tmp_path, m), 4).stdout)
    assert r["constraints_without_evidence"]


def test_blocked_constraint_without_blocker_fails(tmp_path):
    def m(d):
        row = d["matrix_trace"][0]["constraint_trace"][0]
        row["status"] = "blocked"
        row["blocker"] = None
        row["evidence"] = ""
    r = report(run_checker(mutate(tmp_path, m), 4).stdout)
    assert r["blocked_without_reason"]


def test_pass_with_incomplete_trace_fails():
    r = report(run_checker(ROOT / "tests/fixtures/matrix_trace/pass_with_incomplete_matrix_receipt.json", 4).stdout)
    assert "matrix_trace_missing_or_not_array" in r["blockers"]


def test_prose_only_matrix_trace_fails():
    r = report(run_checker(ROOT / "tests/fixtures/matrix_trace/prose_only_matrix_receipt.json", 4).stdout)
    assert "matrix_trace_is_prose_only" in r["blockers"]
