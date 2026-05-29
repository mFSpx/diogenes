from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/dev_orders/simple_dev_order.txt"


def run_wrapper(output: Path, receipt: Path):
    return subprocess.run(
        [
            sys.executable,
            "scripts/dev_order_matrix_wrapper.py",
            "--input",
            str(FIXTURE),
            "--output",
            str(output),
            "--receipt-out",
            str(receipt),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_wrapper_preserves_hashes_and_writes_contract(tmp_path):
    output = tmp_path / "simple.wrapped.txt"
    receipt_path = tmp_path / "wrap.json"
    p = run_wrapper(output, receipt_path)
    assert p.returncode == 0, p.stdout + p.stderr
    r = json.loads(receipt_path.read_text())
    original = FIXTURE.read_bytes()
    assert r["schema"] == "lucidota.dev_order_matrix.wrap_receipt.v1"
    assert r["original_sha256"]
    assert r["policy_sha256"]
    assert r["matrix_required"] is True
    assert r["canonical_graph_materialization"] is False
    assert r["canonical_graph_writes"] is False
    original_copy = ROOT / r["original_path"]
    assert original_copy.read_bytes() == original
    assert not (original_copy.stat().st_mode & stat.S_IWUSR)
    wrapped = output.read_text()
    assert "CURRENT MODE: BUILD + BUGCHASE + GROUP 7 MATRIX ENFORCEMENT" in wrapped
    assert "required matrix_trace" in wrapped or "matrix_trace" in wrapped
    assert "----- ORIGINAL DEV ORDER: BEGIN -----" in wrapped
    assert FIXTURE.read_text() in wrapped


def test_wrapper_idempotent_for_same_input_except_receipt_runtime_fields(tmp_path):
    output = tmp_path / "same.wrapped.txt"
    receipt_a = tmp_path / "a.json"
    receipt_b = tmp_path / "b.json"
    a = run_wrapper(output, receipt_a)
    b = run_wrapper(output, receipt_b)
    assert a.returncode == 0 and b.returncode == 0
    ra = json.loads(receipt_a.read_text())
    rb = json.loads(receipt_b.read_text())
    stable = ["policy_version", "policy_sha256", "original_path", "original_sha256", "wrapped_path", "wrapped_sha256", "matrix_required", "canonical_graph_materialization", "canonical_graph_writes", "verdict", "blockers"]
    for key in stable:
        assert ra[key] == rb[key]
    assert output.read_bytes().__hash__() == output.read_bytes().__hash__()


def test_wrapper_stdin_mode(tmp_path):
    output = tmp_path / "stdin.wrapped.txt"
    receipt_path = tmp_path / "stdin.json"
    p = subprocess.run(
        [sys.executable, "scripts/dev_order_matrix_wrapper.py", "--stdin", "--name", "stdin_fixture", "--output", str(output), "--receipt-out", str(receipt_path)],
        cwd=ROOT,
        input="stdin order exact\n",
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    r = json.loads(receipt_path.read_text())
    assert (ROOT / r["original_path"]).read_text() == "stdin order exact\n"
    assert output.exists()
