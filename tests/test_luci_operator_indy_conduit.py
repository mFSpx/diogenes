from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from scripts import luci_operator as op


def test_operator_indy_trigger_creates_conduit_dry_run_receipt(tmp_path: Path) -> None:
    result = op.process_indy_conduit_for_chat_message(
        "/indy hello from active chat",
        run_id="pytest-indy-dry-run",
        mode="dry-run",
        output_dir=tmp_path,
        database_url="postgresql:///must_not_be_used_in_dry_run",
    )

    assert result["performed"] is True
    assert result["transport"] == "direct_import"
    assert result["executed"] is False
    assert "command" not in result
    receipt = Path(result["receipt_path"])
    assert receipt.exists()
    payload = json.loads(receipt.read_text())
    row = payload["plan"]["dialogue_row"]
    assert row["comms_channel"] == "matrix"
    assert row["sender_id"] == "@luci_operator:local"
    assert row["room_id"] == "!luci_operator:local"
    assert row["event_id"] == result["event_id"]
    assert row["raw_text"] == "/indy hello from active chat"
    assert row["clean_text"] == "/indy hello from active chat"
    assert row["processed_status"] == "queued"


def test_operator_normal_message_is_not_conduit_by_default(tmp_path: Path) -> None:
    result = op.process_indy_conduit_for_chat_message(
        "normal operator chat",
        run_id="pytest-normal-chat",
        mode="off",
        output_dir=tmp_path,
    )

    assert result == {"performed": False, "reason": "not_triggered", "mode": "off"}


def test_luci_operate_payload_includes_indy_conduit_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(op.language_router, "route_text", lambda text, **kwargs: {"intent": "chat", "ontology_terms": [], "lane": {"lane": "FASTLANE"}})
    monkeypatch.setattr(op.language_router, "write", lambda route: {**route, "report_path": "05_OUTPUTS/test_language.json"})
    monkeypatch.setattr(
        op.claw_moa_router,
        "orchestrate_text",
        lambda *args, **kwargs: {
            "input_route": {"lane": "FASTLANE", "route_reason": ["pytest"]},
            "receipt_path": "05_OUTPUTS/test_moa.json",
            "verdict": "PASS",
            "model_calls_performed": False,
            "network_calls_performed": False,
            "model_synthesis": {"performed": False},
            "task_chain": {"enqueue": {}},
            "route_targets": [],
            "lane_plan": {"provider_lanes": {}, "local_model_admission": {}},
        },
    )
    monkeypatch.setattr(op, "run_attempt_engine", lambda *args, **kwargs: {"passed": True, "receipt_path": "05_OUTPUTS/test_attempt.json", "visible_response": {}})
    monkeypatch.setattr(op, "emit_workflow_event", lambda *args, **kwargs: {"performed": True, "event_id": "workflow-event-1"})
    monkeypatch.setattr(op, "append_ingress_cache", lambda entry: "04_RUNTIME/luci/operator_ingress.jsonl")
    monkeypatch.setattr(op, "write_receipt", lambda payload: payload.setdefault("receipt_path", "05_OUTPUTS/luci/test.json"))
    monkeypatch.setattr(op, "compose_response", lambda detail: {"visible_response": {"summary": "Indy_READs: pytest", "next": "next", "segments": []}})

    payload = op.operate(
        "/indy hook me",
        database_url="postgresql:///lucidota_state",
        run_id="pytest-operate-conduit",
        json_out=True,
        conduit_mode="dry-run",
        conduit_output_dir=tmp_path,
    )

    assert payload["indy_conduit"]["performed"] is True
    assert payload["indy_conduit"]["executed"] is False
    assert Path(payload["indy_conduit"]["receipt_path"]).exists()
    assert payload["input"]["indy_conduit"]["performed"] is True
    assert payload["canonical_graph_writes_performed"] is False


def test_luci_shell_exposes_indy_response_via_postgrest_only() -> None:
    text = Path("luci").read_text(encoding="utf-8")
    assert "luci indy-response [--json] [--base-url URL]" in text
    assert "/indy_responses?" in text
    assert "INDY_RESPONSE_API_UNAVAILABLE" in text
    assert "indy_operator_responses.jsonl" not in text


def test_operator_conduit_execute_queues_row_if_db_access_available(tmp_path: Path) -> None:
    psycopg = pytest.importorskip("psycopg")
    database_url = os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    except Exception as exc:
        pytest.skip(f"database unavailable: {type(exc).__name__}: {exc}")

    run_id = "pytest-exec-" + uuid.uuid4().hex
    text = f"/indy pytest execute {run_id}"
    result = op.process_indy_conduit_for_chat_message(
        text,
        run_id=run_id,
        mode="execute",
        output_dir=tmp_path,
        database_url=database_url,
    )

    assert result["performed"] is True
    assert result["executed"] is True
    assert result["db_result"]["dialogue_id"]
    with psycopg.connect(database_url, connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT raw_text, clean_text, processed_status, receipt_id
            FROM ironclaw.waking_dialogue_stream
            WHERE comms_channel = 'matrix' AND event_id = %s;
            """,
            (result["event_id"],),
        )
        row = cur.fetchone()

    assert row == (text, text, "queued", result["receipt_id"])
