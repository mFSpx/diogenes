from __future__ import annotations

import json
from pathlib import Path


def test_fanout_plan_builds_six_lanes_with_exactly_two_vibe_and_two_groq_workers_each() -> None:
    from scripts import recursive_fanout_orchestrator as fanout

    plan = fanout.build_fanout_plan()

    assert len(plan["mini_orchestrators"]) == 6
    for lane in plan["mini_orchestrators"]:
        workers = lane["workers"]
        assert len(workers) == 4
        assert sum(1 for w in workers if w["family"] == "vibe") == 2
        assert sum(1 for w in workers if w["family"] == "groq") == 2
        assert lane["spawn_contract"]["worker_count"] == 4
        assert lane["spawn_contract"]["selection_rule"] == "choose_best_minimal_bundle"
        assert lane["orchestration"]["mode"] == "sub_orchestrator"
        assert lane["orchestration"]["sub_orchestrator_priority"] == ["live_truth_surfaces", "deterministic_local_checks", "thin_packets", "local", "indy_reads", "codex", "vibe", "groq", "broader_cloud"]


def test_workers_reuse_existing_packet_and_dispatch_surfaces() -> None:
    from scripts import recursive_fanout_orchestrator as fanout

    plan = fanout.build_fanout_plan()

    for lane in plan["mini_orchestrators"]:
        for worker in lane["workers"]:
            packet = worker["packet"]
            dispatch_cmd = worker["dispatch_cmd"]
            assert packet["schema"] == "lucidota.goals.agent_packet.v1"
            assert dispatch_cmd[1] == "scripts/goal_swarm_dispatch.py"
            if worker["family"] == "groq":
                assert "scripts/groq_goal_delegate.py" in dispatch_cmd
            else:
                assert packet["target"] == "vibe"


def test_cli_writes_receipt_and_reports_path(tmp_path: Path) -> None:
    from scripts import recursive_fanout_orchestrator as fanout

    receipt = tmp_path / "fanout.json"
    assert fanout.main(["--receipt", str(receipt), "--json"]) == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema"] == "lucidota.recursive_fanout_orchestrator.v1"
    assert payload["mini_orchestrator_count"] == 6
    assert payload["worker_count"] == 24
    assert payload["per_lane_worker_counts"] == [4, 4, 4, 4, 4, 4]
