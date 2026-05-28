from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/system_graph_safety_audit.py", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=2,
    )


def test_graph_safety_help_does_not_execute_audit() -> None:
    proc = run("--help")
    assert proc.returncode == 0
    assert "System-wide graph safety audit" in proc.stdout
    assert "REPORT_PATH=" not in proc.stdout
    assert "SYSTEM_GRAPH_SAFETY=" not in proc.stdout


def test_graph_safety_lists_steps_without_executing() -> None:
    proc = run("--list-steps")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.system_graph_safety_audit.plan.v1"
    assert any("graph_promotion_orphan_detector.py" in step for step in payload["steps"])
    assert any("graph_edge_write_blocker_probe.py --execute" in step for step in payload["steps"])
    assert "REPORT_PATH=" not in proc.stdout
