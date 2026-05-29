from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "activity_tree_ingest_dry_run.py"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, text=True, capture_output=True)


def report_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return ROOT / line.split("=", 1)[1]
    raise AssertionError(stdout)


def test_default_dry_run_writes_pass_report(tmp_path):
    res = run_script("--out-dir", str(tmp_path))
    assert res.returncode == 0, res.stderr
    report = json.loads(report_path(res.stdout).read_text())
    assert report["status"] == "PASS"
    assert report["db_writes_performed"] is False
    assert report["graph_writes_performed"] is False


def test_invalid_json_input_is_receipted_not_traceback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    res = run_script("--input", str(bad), "--out-dir", str(tmp_path / "out"))
    assert res.returncode == 4
    assert "Traceback" not in res.stderr
    report = json.loads(report_path(res.stdout).read_text())
    assert report["status"] == "FAIL"
    assert any(blocker.startswith("input_json_invalid:") for blocker in report["blockers"])


def test_non_object_input_is_receipted(tmp_path):
    bad = tmp_path / "array.json"
    bad.write_text("[]", encoding="utf-8")
    res = run_script("--input", str(bad), "--out-dir", str(tmp_path / "out"))
    assert res.returncode == 4
    report = json.loads(report_path(res.stdout).read_text())
    assert "input_json_must_be_object" in report["blockers"]
