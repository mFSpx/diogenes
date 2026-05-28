import json
import subprocess
import sys
from pathlib import Path

import scripts.goal_agent_packet as goal_agent_packet
from scripts import goal_swarm_dispatch


def test_goal_swarm_dispatch_builds_queue_payloads_and_receipts(tmp_path, monkeypatch):
    calls = []

    class P:
        returncode = 0
        stdout = "REPORT_PATH=05_OUTPUTS/absurd/fake_enqueue.json\n"
        stderr = ""

    def fake_run(cmd, cwd=None, text=None, capture_output=None, check=None):
        calls.append(cmd)
        if cmd[:3] == [sys.executable, "scripts/absurd_queue_spine.py", "--action"]:
            p = Path(goal_swarm_dispatch.ROOT) / "05_OUTPUTS" / "absurd"
            p.mkdir(parents=True, exist_ok=True)
            (p / "fake_enqueue.json").write_text(json.dumps({"action_result": {"job_uuid": "job-1"}, "blockers": []}))
            return P()
        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(goal_agent_packet, "build_packet", lambda **kwargs: {"schema": "lucidota.goals.agent_packet.v1", "adapter": {"selected": "needle_swarm_6x"}, "report_path": None})
    root = Path(goal_swarm_dispatch.ROOT)
    monkeypatch.setattr(sys, "argv", [sys.executable, "--task", "edit code", "--jobs", "2", "--json"])
    assert goal_swarm_dispatch.main() == 0
    payload = json.loads(sorted((root / "05_OUTPUTS" / "goals").glob("goal_swarm_dispatch_*.json"))[-1].read_text())
    assert payload["schema"] == "lucidota.goals.swarm_dispatch.v1"
    assert len(payload["jobs"]) == 2
    assert payload["goal_packet"]["adapter"] == "needle_swarm_6x"
    assert all(job["job_uuid"] == "job-1" for job in payload["jobs"])
    source = Path("scripts/goal_swarm_dispatch.py").read_text().splitlines()
    code_lines = [line for line in source if line.strip() and not line.lstrip().startswith("#")]
    assert len(code_lines) <= 100


def test_goal_swarm_dispatch_detects_groq_command_with_python_prefix(monkeypatch):
    calls = []

    class P:
        returncode = 0
        stdout = "REPORT_PATH=05_OUTPUTS/absurd/fake_groq_enqueue.json\n"
        stderr = ""

    def fake_run(cmd, cwd=None, text=None, capture_output=None, check=None):
        calls.append(cmd)
        p = Path(goal_swarm_dispatch.ROOT) / "05_OUTPUTS" / "absurd"
        p.mkdir(parents=True, exist_ok=True)
        (p / "fake_groq_enqueue.json").write_text(json.dumps({"action_result": {"job_uuid": "job-groq"}, "blockers": []}))
        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(goal_swarm_dispatch, "ensure_queue", lambda name: None)
    monkeypatch.setattr(goal_agent_packet, "build_packet", lambda **kwargs: {"schema": "lucidota.goals.agent_packet.v1", "adapter": {"selected": "needle_swarm_6x"}, "report_path": None})
    monkeypatch.setattr(sys, "argv", [
        sys.executable,
        "--task",
        "audit model lane",
        "--jobs",
        "1",
        "--json",
        "--command",
        sys.executable,
        str(Path(goal_swarm_dispatch.ROOT) / "scripts" / "groq_goal_delegate.py"),
        "--task",
        "audit model lane",
        "--kind",
        "audit",
    ])
    assert goal_swarm_dispatch.main() == 0
    payload = json.loads(sorted((Path(goal_swarm_dispatch.ROOT) / "05_OUTPUTS" / "goals").glob("goal_swarm_dispatch_*.json"))[-1].read_text())
    assert payload["goal_packet"]["adapter"] == "groq"
    assert payload["workflow"] == "goal_swarm"
    assert json.loads(calls[0][calls[0].index("--payload-json") + 1])["selected_adapter"] == "groq"


def test_goal_swarm_dispatch_is_declared_in_recovery_and_policy():
    policy = Path("GOALS/AGENT_ORCHESTRATION_POLICY.md").read_text()
    recovery = Path("scripts/recovery_matrix.py").read_text()
    subsystems = json.loads(Path("GOALS/DEV_MODE_SUBSYSTEMS.json").read_text())
    audit = json.loads(Path("GOALS/DEV_MODE_FEATURE_AUDIT.json").read_text())
    assert "goal_swarm_dispatch" in policy
    assert "goal_swarm_dispatch" in recovery
    assert any(s["id"] == "goal_swarm_dispatch" for s in subsystems["subsystems"])
    assert any(f["key"] == "goal_swarm_dispatch" for f in audit["features"])
