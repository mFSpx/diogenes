import json
import subprocess
import sys
from pathlib import Path

from scripts import goal_swarm_brief


def test_build_swarm_brief_emits_bounded_worker_packets():
    report = goal_swarm_brief.build_swarm_brief()

    assert report["schema"] == "lucidota.goals.swarm_brief.v1"
    assert report["ontology_mode"] == "GO25_STRICT"
    assert report["packets"]
    assert len(report["packets"]) >= 5
    assert report["packets"][0]["output_contract"]["schema"] == "lucidota.worker_order.v1"
    assert report["packets"][0]["output_contract"]["required_output"] == [
        "status",
        "result",
        "next_action",
        "receipt_path",
        "evidence_refs",
        "decision_pairs",
    ]


def test_swarm_brief_cli_writes_a_receipt(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["goal_swarm_brief.py", "--json"])
    rc = goal_swarm_brief.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "GOAL_SWARM_BRIEF=PASS" in out
    assert "lucidota.goals.swarm_brief.v1" in out


def test_swarm_brief_launches_bounded_dispatch_receipts(tmp_path, monkeypatch):
    calls = []

    class P:
        returncode = 0
        stdout = "REPORT_PATH=05_OUTPUTS/goals/fake_dispatch.json\n"
        stderr = ""

    def fake_run(cmd, cwd=None, text=None, capture_output=None, check=None):
        calls.append(cmd)
        if cmd[:3] == [sys.executable, "scripts/goal_swarm_dispatch.py", "--target"]:
            out = Path(goal_swarm_brief.ROOT) / "05_OUTPUTS" / "goals"
            out.mkdir(parents=True, exist_ok=True)
            (out / "fake_dispatch.json").write_text(
                json.dumps({"jobs": [{"job_uuid": "job-1"}], "queue": "goal_swarm", "workflow": "goal_swarm"})
            )
        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["goal_swarm_brief.py", "--launch", "--json"])

    rc = goal_swarm_brief.main()

    assert rc == 0
    assert calls
    assert any(cmd[:3] == [sys.executable, "scripts/goal_swarm_dispatch.py", "--target"] for cmd in calls)
    payload = json.loads(sorted((Path(goal_swarm_brief.ROOT) / "05_OUTPUTS" / "goals").glob("goal_swarm_brief_launch_*.json"))[-1].read_text())
    assert payload["schema"] == "lucidota.goals.swarm_brief_launch.v1"
    assert payload["launch_count"] == payload["packet_count"]
    assert payload["launches"]
    assert payload["launches"][0]["dispatch_report_path"].endswith("fake_dispatch.json")
