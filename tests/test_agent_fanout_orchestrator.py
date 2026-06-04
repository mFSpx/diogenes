from __future__ import annotations

import json
from pathlib import Path


def test_read_current_task_truth_uses_preflight_active_goal(monkeypatch) -> None:
    from scripts import agent_fanout_orchestrator as fanout

    def fake_fetch(path: str, query=None):
        if path == "": return 200, {"paths": {"/active_goal": {"get": {}}, "/flow_receipts": {"get": {}}}}, ""
        if path == "active_goal": return 200, [{"title": "Runner truth from API", "goal": "fanout"}], ""
        if path == "flow_receipts": return 200, [], ""
        raise AssertionError(path)

    monkeypatch.setattr(fanout.preflight, "fetch_json", fake_fetch)
    truth = fanout.read_current_task_truth()
    assert truth["current_task"]["title"] == "Runner truth from API"
    assert truth["openapi_status"] == 200
    assert truth["blockers"] == []


def test_build_plan_emits_six_minis_with_four_vibe_and_two_groq_workers_each() -> None:
    from scripts import agent_fanout_orchestrator as fanout

    plan = fanout.build_plan({"title": "Normalized recursive fanout runner"})
    assert plan["mini_orchestrator_count"] == 6
    assert plan["worker_count"] == 36
    assert plan["per_lane_worker_counts"] == [6, 6, 6, 6, 6, 6]
    for lane in plan["mini_orchestrators"]:
        workers = lane["workers"]
        assert sum(1 for w in workers if w["family"] == "vibe") == 4
        assert sum(1 for w in workers if w["family"] == "groq") == 2
        assert lane["spawn_contract"]["selection_rule"] == "choose_best_minimal_bundle"
        groq_cmds = [w["dispatch_cmd"] for w in workers if w["family"] == "groq"]
        assert all("scripts/groq_goal_delegate.py" in cmd for cmd in groq_cmds)


def test_worker_return_check_rejects_commentary_only_text() -> None:
    from scripts import agent_fanout_orchestrator as fanout

    bad = fanout.worker_return_check("looks good; maybe run pytest")
    good = fanout.worker_return_check({
        "status": "ok",
        "result": {"changed_files": ["scripts/agent_fanout_orchestrator.py"]},
        "next_action": "run focused tests",
        "receipt_path": "05_OUTPUTS/goals/x.json",
        "evidence_refs": ["tests/test_agent_fanout_orchestrator.py"],
        "decision_pairs": [["adapter", "groq"], ["verdict", "accept"]],
    })
    assert bad == {"accepted": False, "reason": "commentary_only_worker_return", "detail": "non_json_text"}
    assert good["accepted"] is True


def test_main_blocks_and_writes_json_when_db_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    from scripts import agent_fanout_orchestrator as fanout

    monkeypatch.setattr(fanout, "read_current_task_truth", lambda: {
        "schema": "lucidota.agent_fanout_preflight.v1",
        "postgrest_base_url": "http://127.0.0.1:3000",
        "openapi_status": 200,
        "openapi_error": "",
        "active_goal_status": 200,
        "active_goal_error": "",
        "current_task": {"title": "Runner lane"},
        "route_findings": [],
        "blockers": [],
    })
    monkeypatch.setattr(fanout, "write_db_receipt", lambda payload: (_ for _ in ()).throw(RuntimeError("db down")))
    receipt = tmp_path / "agent_fanout.json"
    assert fanout.main(["--receipt", str(receipt), "--json"]) == 2
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema"] == "lucidota.agent_fanout_orchestrator.v1"
    assert payload["status"] == "blocked"
    assert "DB_BLOCKED" in payload["blockers"]
