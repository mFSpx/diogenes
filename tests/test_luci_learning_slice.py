from __future__ import annotations

from pathlib import Path

from scripts.luci_learning_slice import classify_candidate, map_board_state


def test_board_state_map_and_candidate_class(tmp_path: Path):
    artifact = tmp_path / "dev_journey_decision_points.py"
    artifact.write_text("print('treelite router artifact')\n", encoding="utf-8")

    board = map_board_state(
        "study one current source or internal artifact, extract one reusable improvement, test it, and receipt the result",
        artifact,
    )
    candidate = classify_candidate(artifact, "study one current source or internal artifact, extract one reusable improvement, test it, and receipt the result")

    assert "operator" in board.actors
    assert "LUCI" in board.actors
    assert artifact.name in board.actors
    assert "receipt law" in board.constraints
    assert "existing harness" in board.leverage
    assert "treelite routers" in board.resources
    assert candidate["candidate_kind"] == "algorithm_trial_harness"
    assert candidate["candidate_name"] == "luci_algorithm_trial_harness"
    assert "board-state classification" in candidate["feature_hypothesis"]
