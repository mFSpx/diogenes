from __future__ import annotations

def test_absurd_gate_degrades_on_nonhard_model_audit_failure() -> None:
    from scripts.absurd_gate import combine_gate_verdict
    verdict = combine_gate_verdict([
        {'name':'ledger','verdict':'PASS'},
        {'name':'model_audit','verdict':'FAIL','hard':False},
    ], hard_failures=[])
    assert verdict == 'DEGRADED'

def test_absurd_gate_reads_child_verdict_markers() -> None:
    from scripts.absurd_gate import child_verdict
    assert child_verdict('absurd_health_check', 0, 'ABSURD_HEALTH_CHECK=DEGRADED\n') == 'DEGRADED'
    assert child_verdict('model_audit_absurd_adapter', 4, 'MODEL_AUDIT_ABSURD_ADAPTER=FAIL\n') == 'DEGRADED'

def test_absurd_operator_next_move_does_not_claim_model_repair_when_all_checks_pass() -> None:
    from scripts.absurd_gate import operator_next_smallest_safe_work
    assert operator_next_smallest_safe_work('PASS', [{'name': 'model_audit_absurd_adapter', 'verdict': 'PASS'}]) == 'run next-move #1'
