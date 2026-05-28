#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from absurd_queue_spine import MAX_PAYLOAD_JSON_BYTES, load_payload_json, run_job


def report_from(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def test_load_payload_json_requires_object_and_byte_bound():
    assert load_payload_json('{"ok": true}') == {"ok": True}
    try:
        load_payload_json("[]")
    except ValueError as exc:
        assert str(exc) == "payload_json_must_be_object"
    else:
        raise AssertionError("expected non-object payload rejection")
    try:
        load_payload_json('{"x":"' + ("a" * MAX_PAYLOAD_JSON_BYTES) + '"}')
    except ValueError as exc:
        assert str(exc).startswith("payload_json_too_large:")
    else:
        raise AssertionError("expected oversized payload rejection")


def test_enqueue_invalid_payload_writes_blocked_receipt_without_db():
    proc = subprocess.run(
        [sys.executable, "scripts/absurd_queue_spine.py", "--action", "enqueue", "--payload-json", "not-json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    report = report_from(proc.stdout)
    assert report["db_writes_performed"] is False
    assert report["canonical_graph_writes_performed"] is False
    assert report["blockers"] == ["payload_json_invalid"]
    assert report["action_result"]["execute_performed"] is False
    assert report["action_result"]["payload_sha256"] is None


def test_enqueue_non_object_payload_writes_specific_blocker():
    proc = subprocess.run(
        [sys.executable, "scripts/absurd_queue_spine.py", "--action", "enqueue", "--payload-json", "[]"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 1
    report = report_from(proc.stdout)
    assert report["blockers"] == ["payload_json_must_be_object"]


def test_external_command_job_runs_allowlisted_python_script(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "REPORT_PATH=05_OUTPUTS/goals/fake.json\n"
        stderr = ""

    seen = {}

    def fake_run(cmd, cwd=None, text=None, capture_output=None, timeout=None, check=None):
        seen["cmd"] = cmd
        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, result, err = run_job(
        "external_command",
        {
            "command": [sys.executable, "scripts/goal_agent_packet.py", "--target", "generic", "--task", "recovery", "--complexity", "simple", "--json"],
        },
    )
    assert ok is True
    assert err == ""
    assert seen["cmd"][1] == "scripts/goal_agent_packet.py"
    assert "goal_agent_packet.py" in result["command"]


def test_usecase_proof_is_absurd_allowlisted():
    import absurd_queue_spine
    assert "scripts/lucidota_usecase_proof.py" in absurd_queue_spine.ALLOWED_EXTERNAL_COMMANDS
