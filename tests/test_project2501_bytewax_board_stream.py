from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def sample_rows() -> list[dict]:
    return [
        {
            "event_id": "a" * 64,
            "actor": "operator",
            "source": "operator_chat",
            "lane": "audit",
            "route_key": "audit:operator_chat:operator",
            "verdict": "win",
            "event_created_at": "2026-01-01T00:00:00Z",
            "board_features": {"token_count": 40, "needs_graph_write": True, "risk_of_slop": 0.22},
            "route_cost": {"tokens": 40, "vram": 0},
            "gain": {"reduced_entropy": 0.5, "routing_accuracy": 0.82},
            "receipt": "05_OUTPUTS/project2501_board_moves/x.json",
        }
    ]


def test_schema_defines_board_stream_cursor_run_hint_and_no_ui_feature_change() -> None:
    schema = (ROOT / "06_SCHEMA" / "113_project2501_bytewax_board_stream.sql").read_text(encoding="utf-8")
    assert "lucidota_control.board_stream_cursor" in schema
    assert "lucidota_control.board_stream_run" in schema
    assert "lucidota_control.board_stream_hint" in schema
    assert "lucidota_control.watch_metric" in schema
    assert "operator_feature_authority_required" in schema
    assert "canonical_graph_writes_performed boolean NOT NULL DEFAULT false" in schema
    stream_source = (ROOT / "scripts" / "project2501_bytewax_board_stream.py").read_text(encoding="utf-8")
    assert "116_operator_feedback_signal.sql" in stream_source
    assert "fn_train_operator_feedback_batch" in stream_source


def test_flow_turns_board_rows_into_stream_hints() -> None:
    from project2501_bytewax_board_stream import flow

    hints = flow(sample_rows())

    assert len(hints) == 1
    hint = hints[0]
    assert hint["schema"] == "lucidota.project2501.board_stream_hint.v1"
    assert hint["event_id"] == "a" * 64
    assert hint["lane"] == "audit"
    assert hint["actor"] == "operator"
    assert hint["score"] >= 0
    assert hint["canonical_graph_writes_performed"] is False
    assert hint["detail"]["bytewax_flow"] == "project2501_board_stream"
    assert "treelite" in hint["detail"]
    assert hint["detail"]["ternary_logic"]["valency"] == 1
    assert hint["detail"]["ternary_logic"]["net_spatial_polarity"] == 1
    assert hint["detail"]["ternary_logic"]["corridor_state"] == "signal"
    assert hint["detail"]["ternary_logic"]["notification_suppressed"] is False


def test_flow_accumulates_balanced_ternary_corridor_state() -> None:
    from project2501_bytewax_board_stream import flow

    rows = sample_rows()
    rows.append(
        {
            **sample_rows()[0],
            "event_id": "b" * 64,
            "verdict": "failed",
            "receipt": "05_OUTPUTS/project2501_board_moves/y.json",
        }
    )

    hints = flow(rows)

    assert [h["detail"]["ternary_logic"]["valency"] for h in hints] == [1, -1]
    assert [h["detail"]["ternary_logic"]["net_spatial_polarity"] for h in hints] == [1, 0]
    assert hints[1]["detail"]["ternary_logic"]["kleene_k3_state"] == -1
    assert hints[1]["detail"]["ternary_logic"]["corridor_state"] == "stasis"
    assert hints[1]["detail"]["ternary_logic"]["notification_suppressed"] is True


def test_cli_dry_run_writes_receipt_without_db_writes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/project2501_bytewax_board_stream.py", "once", "--limit", "3", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PROJECT2501_BYTEWAX_BOARD_STREAM=PASS" in proc.stdout
    report_path = next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    assert receipt["execute_performed"] is False
    assert receipt["canonical_graph_writes_performed"] is False
    assert receipt["status"] == "PASS"
    assert receipt["mode"] == "latest_window"
