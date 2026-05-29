from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_core_contract_doc_is_active_law() -> None:
    doc = (ROOT / "00_PROJECT_BRAIN" / "PROJECT_2501_CORE_CONTRACT.md").read_text(encoding="utf-8")
    required = [
        "Every input becomes an EventEnvelope.",
        "Every EventEnvelope is timestamped, hashed, embedded, classified, and routed.",
        "deterministic rules first, Treelite second, model fallback last",
        "Every durable action emits a WorkReceipt.",
        "Every WorkReceipt updates River training rows.",
        "Every graph write is staged, justified, receipt-backed, and replayable.",
        "Every UI tile reads from receipts, not claims.",
        "Big Board feature additions/removals are operator-authority changes.",
        "Ledger first, screen second, siempre.",
    ]
    for line in required:
        assert line in doc
    index = (ROOT / "00_PROJECT_BRAIN" / "ACTIVE_INSTRUCTION_INDEX.md").read_text(encoding="utf-8")
    assert "00_PROJECT_BRAIN/PROJECT_2501_CORE_CONTRACT.md" in index
    registry = json.loads((ROOT / "00_PROJECT_BRAIN" / "instruction_authority_registry.json").read_text(encoding="utf-8"))
    assert "00_PROJECT_BRAIN/PROJECT_2501_CORE_CONTRACT.md" in registry["canonical_files"]
    assert any(law["law_key"] == "project2501_core_contract" for law in registry["active_laws"])
    assert any(law["law_key"] == "big_board_operator_metric_ledger" for law in registry["active_laws"])


def test_core_board_schema_defines_minimum_board_tables_and_no_direct_graph_write() -> None:
    schema = (ROOT / "06_SCHEMA" / "112_project2501_core_board.sql").read_text(encoding="utf-8")
    for table in [
        "event_envelope",
        "raw_artifact",
        "work_order",
        "work_receipt",
        "model_invocation",
        "route_decision",
        "board_position",
        "river_training_row",
        "treelite_gate_version",
        "script_manifest",
        "corpse_manifest",
        "dead_letter",
        "watch_metric",
    ]:
        assert f"lucidota_control.{table}" in schema
    assert "canonical_graph_writes_performed boolean NOT NULL DEFAULT false" in schema
    assert "graph_write_mode text NOT NULL DEFAULT 'staged_only'" in schema


def test_board_move_pipeline_builds_event_route_receipt_and_river_row() -> None:
    from project2501_board_move import build_board_move

    result = build_board_move(
        actor="operator",
        source="operator_chat",
        text="Patch the repo, verify tests, stage graph write only with receipt.",
        execute=False,
        position="pytest-position",
    )

    envelope = result["event_envelope"]
    decision = result["route_decision"]
    receipt = result["work_receipt"]
    river = result["river_training_row"]
    assert envelope["schema"] == "lucidota.project2501.event_envelope.v1"
    assert len(envelope["event_id"]) == 64
    assert envelope["actor"] == "operator"
    assert envelope["verbatim_hash"]
    assert envelope["board_features"]["mutation_requested"] is True
    assert decision["schema"] == "lucidota.project2501.route_decision.v1"
    assert decision["gate_order"] == ["deterministic_rules", "treelite_gate", "model_fallback"]
    assert decision["model_fallback"]["used"] is False
    assert decision["lane"] in {"slow", "audit"}
    assert decision["treelite_gate"]["available"] is True
    assert receipt["schema"] == "lucidota.project2501.work_receipt.v1"
    assert receipt["canonical_graph_writes_performed"] is False
    assert river["schema"] == "lucidota.project2501.river_training_row.v1"
    assert river["event_id"] == envelope["event_id"]
    assert river["route_chosen"] == decision["lane"]


def test_board_move_cli_dry_run_writes_receipt() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/project2501_board_move.py",
            "ingest",
            "--actor",
            "operator",
            "--source",
            "operator_chat",
            "--text",
            "status ping receipt",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PROJECT2501_BOARD_MOVE=PASS" in proc.stdout
    report_path = next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["execute_performed"] is False
    assert receipt["event_envelope"]["source"] == "operator_chat"
    assert receipt["route_decision"]["lane"] == "fast"
