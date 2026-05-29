from __future__ import annotations


def test_next_move_engine_prioritizes_model_audit_gap() -> None:
    from scripts.abductive_next_move_engine import generate_next_moves

    board = {
        "open_blockers": [{"class": "model_audit", "severity": 100, "summary": "missing audit"}],
        "object_counts": {"GraphCandidate": 93},
    }
    moves = generate_next_moves(board)
    assert moves
    assert moves[0]["class"] == "model_audit"
    assert moves[0]["requires_operator_approval"] is False
    assert moves[0]["expected_gain"] >= moves[-1]["expected_gain"]

