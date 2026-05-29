from __future__ import annotations

def test_absurd_next_move_prioritizes_model_audit_gap() -> None:
    from scripts.absurd_next_move_engine import generate_next_moves
    board = {'open_blockers':[{'class':'model_audit','severity':95,'summary':'missing valid audit'}], 'object_counts':{'GraphCandidate':93}}
    moves = generate_next_moves(board)
    assert len(moves) == 5
    assert moves[0]['class'] == 'model_audit'
    assert moves[0]['requires_operator_approval'] is False
    assert moves[0]['expected_gain'] >= moves[-1]['expected_gain']
    assert any(
        'scripts/bytewax_absurd_stream_audit.py' in cmd
        for move in moves
        for cmd in move['commands']
    )
