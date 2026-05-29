#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_goal_model_fabric_orchestrator_dry_run_has_required_postgres_jobs():
    proc = subprocess.run([sys.executable, "scripts/goal_model_fabric_orchestrate.py", "--json"], cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    commands = [" ".join(j["command"]) for j in payload["planned_jobs"]]
    assert any("goal_model_fabric_control.py start" in c and "--target heavy" in c for c in commands)
    assert any("lucidota_model_turbine_overseer.py --assign" in c for c in commands)
    assert any("groq_goal_delegate.py" in c for c in commands)
    assert payload["ontology"]["official_ontology"] == "GO-25"
    assert payload["execute_performed"] is False
