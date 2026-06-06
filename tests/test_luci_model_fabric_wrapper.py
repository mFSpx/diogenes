#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_model_fabric_orchestrator():
    proc = subprocess.run(
        [str(ROOT / "luci"), "model", "fabric", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    assert payload["schema"] == "lucidota.goals.model_fabric_orchestrate.v1"
    assert any(job["name"] == "gemini_provider_execute" for job in payload["planned_jobs"])
    assert any(job["name"] == "vibe_lane_execute" for job in payload["planned_jobs"])
    assert payload["execute_performed"] is False
