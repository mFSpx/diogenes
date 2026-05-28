from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


def write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def iso(base: datetime, hours: float) -> str:
    return (base + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def seed_required_lanes(root: Path, *, span_hours: float = 5.0, canonical_write: bool = False) -> None:
    base = datetime(2026, 5, 27, 0, 0, tzinfo=timezone.utc)
    receipts = [
        ("05_OUTPUTS/project2501_board_stream/project2501_bytewax_board_stream_once_execute_a.json", {"schema":"x", "generated_at":iso(base,0), "status":"PASS", "canonical_graph_writes_performed":False}),
        ("05_OUTPUTS/project2501_board_stream/project2501_bytewax_board_stream_once_execute_b.json", {"schema":"x", "generated_at":iso(base,span_hours), "status":"PASS", "canonical_graph_writes_performed":canonical_write}),
        ("05_OUTPUTS/abductive_db_os/abductive_db_os_gate_fast_a.json", {"schema":"x", "started_at_utc":iso(base,0), "verdict":"PASS", "canonical_graph_writes":False, "canonical_graph_materialization":False}),
        ("05_OUTPUTS/abductive_db_os/abductive_db_os_gate_fast_b.json", {"schema":"x", "started_at_utc":iso(base,span_hours), "verdict":"PASS", "canonical_graph_writes":False, "canonical_graph_materialization":False}),
        ("05_OUTPUTS/absurd_abductive/absurd_gate_fast_a.json", {"schema":"x", "started_at_utc":iso(base,0), "verdict":"PASS", "canonical_graph_writes":False, "canonical_graph_materialization":False}),
        ("05_OUTPUTS/absurd_abductive/absurd_gate_fast_b.json", {"schema":"x", "started_at_utc":iso(base,span_hours), "verdict":"PASS", "canonical_graph_writes":False, "canonical_graph_materialization":False}),
        ("05_OUTPUTS/goals/swarm_usage_ledger_a.json", {"schema":"lucidota.goals.swarm_usage_ledger.v1", "generated_at":iso(base,0), "totals":{"all_accounted_tokens":10}, "target_policy":{"local":{"target_share":0.4}}}),
        ("05_OUTPUTS/goals/swarm_usage_ledger_b.json", {"schema":"lucidota.goals.swarm_usage_ledger.v1", "generated_at":iso(base,span_hours), "totals":{"all_accounted_tokens":20}, "target_policy":{"local":{"target_share":0.4}}}),
        ("05_OUTPUTS/slop_audit/slop_audit_law_a.json", {"schema":"x", "generated_at":iso(base,0), "status":"PASS", "blockers":[]}),
        ("05_OUTPUTS/slop_audit/slop_audit_law_b.json", {"schema":"x", "generated_at":iso(base,span_hours), "status":"PASS", "blockers":[]}),
        ("05_OUTPUTS/graph/graph_promotion_gate_execute_a.json", {"schema":"x", "generated_at":iso(base,span_hours), "db_writes_performed":True, "canonical_graph_writes_performed":False, "graph_writes_performed":False, "blockers":[]}),
    ]
    for rel, payload in receipts:
        write(root / rel, payload)


def test_full_system_soak_passes_when_required_receipts_span_minimum(tmp_path):
    from scripts.full_system_soak_audit import build_audit

    seed_required_lanes(tmp_path, span_hours=5.25)
    audit = build_audit(root=tmp_path, min_hours=4.0)

    assert audit["status"] == "PASS"
    assert audit["full_system_soak_passed"] is True
    assert audit["min_hours"] == 4.0
    assert audit["lanes"]["bytewax_stream"]["span_hours"] >= 5.25
    assert audit["invariants"]["canonical_graph_writes_absent"]["passed"] is True


def test_full_system_soak_degrades_when_only_one_lane_lacks_span(tmp_path):
    from scripts.full_system_soak_audit import build_audit

    seed_required_lanes(tmp_path, span_hours=5.0)
    for path in (tmp_path / "05_OUTPUTS/abductive_db_os").glob("abductive_db_os_gate_fast_b.json"):
        path.unlink()
    audit = build_audit(root=tmp_path, min_hours=4.0)

    assert audit["status"] == "DEGRADED"
    assert audit["full_system_soak_passed"] is False
    assert "dbos_gate:span_below_min_hours" in audit["blockers"]
    assert audit["lanes"]["bytewax_stream"]["passed"] is True


def test_full_system_soak_fails_on_canonical_graph_write_evidence(tmp_path):
    from scripts.full_system_soak_audit import build_audit

    seed_required_lanes(tmp_path, span_hours=5.0, canonical_write=True)
    audit = build_audit(root=tmp_path, min_hours=4.0)

    assert audit["status"] == "FAIL"
    assert audit["full_system_soak_passed"] is False
    assert audit["invariants"]["canonical_graph_writes_absent"]["passed"] is False


def test_full_system_soak_ignores_bad_receipts_before_current_window(tmp_path):
    from scripts.full_system_soak_audit import build_audit

    seed_required_lanes(tmp_path, span_hours=5.0)
    old = {
        "schema": "x",
        "generated_at": "2026-05-26T00:00:00Z",
        "status": "FAIL",
        "blockers": ["old_slop_before_current_window"],
    }
    write(tmp_path / "05_OUTPUTS/slop_audit/slop_audit_law_old_fail.json", old)

    audit = build_audit(root=tmp_path, min_hours=4.0)

    assert audit["status"] == "PASS"
    assert audit["lanes"]["slop_audit"]["bad_receipt_count"] == 1
    assert audit["lanes"]["slop_audit"]["recent_bad_receipt_count"] == 0
