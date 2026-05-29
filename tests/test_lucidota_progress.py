import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lucidota_progress_falls_back_to_status_ledger_when_build_plan_missing():
    assert not (ROOT / "00_PROJECT_BRAIN" / "BUILD_PLAN_AUDIT.md").exists()
    proc = subprocess.run(
        [sys.executable, "scripts/lucidota_progress.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Source: status ledger" in proc.stdout
    assert "Overall:" in proc.stdout
    assert "Open blockers:" in proc.stdout


def test_lucidota_progress_has_no_traceback_on_missing_legacy_plan():
    proc = subprocess.run(
        [sys.executable, "scripts/lucidota_progress.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert "Traceback" not in proc.stdout + proc.stderr
