from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def sample_work_order() -> dict:
    return {
        "work_order_uuid": "11111111-1111-4111-8111-111111111111",
        "event_id": "a" * 64,
        "decision_uuid": "22222222-2222-4222-8222-222222222222",
        "lane": "audit",
        "work_kind": "bounded_audit",
        "status": "queued",
        "payload": {"raw_ref": "inline://x", "route_key": "audit:operator_chat:operator"},
        "idempotency_key": "idem-test",
    }


def sample_event() -> dict:
    return {
        "event_id": "a" * 64,
        "actor": "operator",
        "source": "operator_chat",
        "raw_ref": "inline://x",
        "text": "Verify staged board work order and emit receipt.",
        "board_features": {
            "token_count": 12,
            "needs_graph_write": True,
            "expected_gain": 0.76,
            "risk_of_slop": 0.2,
        },
    }


def sample_decision() -> dict:
    return {
        "decision_uuid": "22222222-2222-4222-8222-222222222222",
        "lane": "audit",
        "route_key": "audit:operator_chat:operator",
        "expected_gain": 0.76,
        "confidence": 0.94,
        "graph_write_mode": "staged_only",
        "cost": {"tokens": 12, "vram": 0, "risk": 0.2},
    }


def test_worker_schema_defines_claim_run_and_no_graph_writes() -> None:
    schema = (ROOT / "06_SCHEMA" / "114_project2501_board_worker.sql").read_text(encoding="utf-8")
    assert "lucidota_control.board_worker_run" in schema
    assert "idx_project2501_work_order_claim" in schema
    assert "canonical_graph_writes_performed boolean NOT NULL DEFAULT false" in schema
    assert "operator_feature_authority_required boolean NOT NULL DEFAULT true" in schema


def test_build_worker_result_emits_receipt_river_and_metric_without_graph_write() -> None:
    from project2501_board_worker import build_worker_result

    result = build_worker_result(
        work_order=sample_work_order(),
        event=sample_event(),
        decision=sample_decision(),
        worker_id="pytest-worker",
        execute=True,
        latency_ms=12.5,
        receipt_path="05_OUTPUTS/project2501_board_worker/x.json",
    )

    receipt = result["work_receipt"]
    river = result["river_training_row"]
    metric = result["watch_metric"]
    run = result["board_worker_run"]
    assert receipt["schema"] == "lucidota.project2501.board_worker.work_receipt.v1"
    assert receipt["verdict"] == "win"
    assert receipt["canonical_graph_writes_performed"] is False
    assert receipt["graph_write_mode"] == "staged_only"
    assert receipt["artifact_refs"] == ["inline://x", "06_SCHEMA/114_project2501_board_worker.sql"]
    assert river["route_chosen"] == "audit"
    assert river["model_used"] == "none"
    assert river["tokens_in"] == 12
    assert metric["metric_key"] == "project2501_board_worker_once"
    assert metric["operator_feature_authority_required"] is True
    assert run["status"] == "succeeded"
    assert run["bounded_step"] == "validate_and_receipt"


def test_worker_source_uses_skip_locked_and_claims_one_bounded_work_order() -> None:
    source = (ROOT / "scripts" / "project2501_board_worker.py").read_text(encoding="utf-8")
    assert "FOR UPDATE SKIP LOCKED" in source
    assert "LIMIT 1" in source
    assert "status IN ('created','queued')" in source
    assert "canonical_graph_writes_performed" in source


def test_claim_sql_locks_work_order_before_nullable_route_join() -> None:
    from project2501_board_worker import claim_sql

    sql, params = claim_sql(["audit", "slow"])

    assert params == ["audit", "slow"]
    assert "WITH claimed AS" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert sql.index("FOR UPDATE SKIP LOCKED") < sql.index("LEFT JOIN lucidota_control.route_decision")


def test_worker_cli_dry_run_writes_receipt_without_db_writes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/project2501_board_worker.py", "worker-once", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PROJECT2501_BOARD_WORKER=PASS" in proc.stdout
    report_path = next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["execute_performed"] is False
    assert receipt["canonical_graph_writes_performed"] is False
