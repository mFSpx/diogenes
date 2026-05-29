#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def receipt_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            return ROOT / line.split("=", 1)[1]
    raise AssertionError(stdout)


def make_reference(tmp_path: Path, lines: int = 10) -> Path:
    ref = tmp_path / "pocketflow_ref.py"
    ref.write_text("\n".join(f"# line {i}" for i in range(lines)), encoding="utf-8")
    return ref


def make_large_python(tmp_path: Path, body_lines: int = 60) -> Path:
    target = tmp_path / "target.py"
    body = "\n".join("    x = 1" for _ in range(body_lines))
    target.write_text(f"def oversized():\n{body}\n    return x\n", encoding="utf-8")
    return target


def test_slop_audit_warns_on_5x_pocketflow_without_failing(tmp_path: Path) -> None:
    ref = make_reference(tmp_path, 10)
    target = make_large_python(tmp_path, 60)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/slop_audit_law.py",
            "--paths",
            str(target),
            "--pocketflow-file",
            str(ref),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SLOP_AUDIT_LAW=REVIEW" in proc.stdout
    data = json.loads(receipt_path(proc.stdout).read_text(encoding="utf-8"))
    assert data["schema"] == "lucidota.slop_audit_law.v1"
    assert data["model_calls_performed"] is False
    assert data["local_law"]["workflow_path_authority"] == "source_code_blueprint"
    assert any(item["tier"] == "over_5x_pocketflow" for item in data["warnings"])


def test_slop_audit_strict_turns_complexity_review_into_blocker(tmp_path: Path) -> None:
    ref = make_reference(tmp_path, 10)
    target = make_large_python(tmp_path, 60)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/slop_audit_law.py",
            "--paths",
            str(target),
            "--pocketflow-file",
            str(ref),
            "--strict",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 4
    data = json.loads(receipt_path(proc.stdout).read_text(encoding="utf-8"))
    assert data["verdict"] == "FAIL"
    assert any(item["kind"] == "code_span_complexity_review" for item in data["blockers"])


def test_slop_audit_help_is_runnable() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/slop_audit_law.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0
    assert "PocketFlow" in proc.stdout
