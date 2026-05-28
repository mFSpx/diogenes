from __future__ import annotations

from pathlib import Path


def test_gate_verdict_degrades_on_model_audit_failure_without_graph_writes() -> None:
    from scripts.abductive_db_os_gate import combine_gate_verdict

    verdict = combine_gate_verdict(
        checks=[
            {"name": "board_state", "verdict": "PASS"},
            {"name": "model_audit", "verdict": "FAIL", "hard": False},
            {"name": "canonical_graph_writes", "verdict": "PASS"},
        ],
        hard_failures=[],
    )
    assert verdict == "DEGRADED"


def test_gate_command_plan_references_existing_local_scripts() -> None:
    from scripts.abductive_db_os_gate import command_plan

    root = Path(__file__).resolve().parents[1]
    missing = [cmd[1] for cmd in command_plan("fast") if cmd[1].startswith("scripts/") and not (root / cmd[1]).exists()]
    assert missing == []


def test_gate_child_verdict_reads_db_os_markers() -> None:
    from scripts.abductive_db_os_gate import child_verdict

    assert child_verdict("model_audit_db_adapter", 4, "MODEL_AUDIT_DB_ADAPTER=FAIL") == "DEGRADED"
    assert child_verdict("abductive_db_os_health_check", 4, "ABDUCTIVE_DB_OS_HEALTH_CHECK=FAIL") == "FAIL"
    assert child_verdict("bytewax_db_os_stream_audit", 0, "BYTEWAX_DB_OS_STREAM_AUDIT=DEGRADED") == "DEGRADED"


def test_operator_next_move_does_not_claim_model_repair_when_all_checks_pass() -> None:
    from scripts.abductive_db_os_gate import operator_next_smallest_safe_work

    checks = [
        {"name": "model_audit_db_adapter", "verdict": "PASS"},
        {"name": "abductive_next_move_engine", "verdict": "PASS"},
    ]

    assert operator_next_smallest_safe_work("PASS", checks) == "run next-move #1"


def test_operator_next_move_names_model_repair_only_when_model_audit_degrades() -> None:
    from scripts.abductive_db_os_gate import operator_next_smallest_safe_work

    checks = [{"name": "model_audit_db_adapter", "verdict": "DEGRADED"}]

    assert operator_next_smallest_safe_work("DEGRADED", checks) == "repair model audit block 0001"
