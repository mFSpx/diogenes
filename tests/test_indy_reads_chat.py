from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.indy_reads as indy_reads


class _Cursor:
    def __init__(self):
        self.executed: list[tuple[str, tuple | None]] = []
        self.fetchone_result = ("job-uuid-1", True)

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Conn:
    def __init__(self):
        self.cursor_obj = _Cursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_load_next_goal_queue_reads_goals_queue(tmp_path: Path, monkeypatch) -> None:
    goals_queue = tmp_path / "GOALS" / "NEXT_GOAL_QUEUE.json"
    goals_queue.parent.mkdir(parents=True)
    goals_queue.write_text(
        json.dumps({"schema": "lucidota.goals.next_goal_queue.v1", "queue": [{"title": "Alpha"}, {"title": "Beta"}]}),
        encoding="utf-8",
    )
    handoff = tmp_path / "GOALS" / "CURRENT_HANDOFF.md"
    handoff.write_text("# CURRENT GOAL HANDOFF\n\nSave This Prompt, Pass on this Handoff:\n", encoding="utf-8")
    monkeypatch.setattr(indy_reads, "GOALS_NEXT_GOAL_QUEUE", goals_queue)
    monkeypatch.setattr(indy_reads, "GOALS_HANDOFF_MD", handoff)

    orders = indy_reads.load_next_goal_queue()

    assert [order["title"] for order in orders] == ["Alpha", "Beta"]
    assert "Save This Prompt, Pass on this Handoff:" in indy_reads.load_goals_handoff_text()


def test_enqueue_goal_work_order_writes_absurd_queue_job(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(indy_reads.psycopg, "connect", lambda dsn: conn)

    result = indy_reads.enqueue_goal_work_order(
        {
            "order_id": "goal-chat-1",
            "queue": "control",
            "workflow": "goal_swarm_dispatch",
            "job_kind": "external_command",
            "payload": {"command": [".venv/bin/python", "scripts/goal_swarm_dispatch.py", "--target", "generic", "--task", "demo", "--jobs", "1", "--json"]},
        }
    )

    assert result["ok"] is True
    assert result["job_uuid"] == "job-uuid-1"
    assert conn.committed is True
    assert any("absurd_queue_job" in sql for sql, _ in conn.cursor_obj.executed)
    assert any("absurd_queue_event" in sql for sql, _ in conn.cursor_obj.executed)


def test_load_queued_conduit_dialogue_reads_matrix_rows_for_indy_reads(monkeypatch):
    rows = [{"event_id": "$evt", "raw_text": " raw ", "clean_text": "raw", "sender_id": "@op"}]

    class _ReadConn:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(indy_reads.psycopg, "connect", lambda dsn: _ReadConn())
    import indy_conduit_driver
    monkeypatch.setattr(indy_conduit_driver, "read_queued_dialogue_rows", lambda conn, limit=5: rows)

    assert indy_reads.load_queued_conduit_dialogue(limit=5) == rows
    assert "Indy_READs" in Path(indy_reads.__file__).read_text()


def test_format_conduit_dialogue_row_uses_clean_text_for_operator_surface():
    rendered = indy_reads.format_conduit_dialogue_row(
        {"event_id": "$evt", "raw_text": "  noisy raw  ", "clean_text": "clean line", "sender_id": "@op"},
        1,
    )

    assert rendered == "1. @op $evt: clean line"


def test_queued_dialogue_context_uses_waking_dialogue_contract():
    row = {
        "id": "dialogue-1",
        "event_id": "$evt",
        "raw_text": "  raw text  ",
        "clean_text": "raw text",
        "extracted_entities": {"hashtags": ["Smoke"], "slash_commands": ["flow"]},
    }

    book, page, parser = indy_reads.queued_dialogue_context(row)

    assert book.id == "waking_dialogue::dialogue-1"
    assert book.name == "ironclaw.waking_dialogue_stream"
    assert page["text"] == "raw text"
    assert page["extract_method"] == "waking_dialogue_stream"
    assert parser["parser_version"] == "waking_dialogue_chat_v1"
    assert parser["terms"] == ["Smoke", "flow"]


def test_record_conduit_dialogue_response_records_comment_not_network_send(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_record(**kwargs):
        calls["record"] = kwargs

    def fake_tune(*args, **kwargs):
        calls["tune"] = {"args": args, "kwargs": kwargs}
        return {}

    monkeypatch.setattr(indy_reads, "record_indy_judgment", fake_record)
    monkeypatch.setattr(indy_reads, "tune_and_record_heartbeat", fake_tune)
    monkeypatch.setattr(indy_reads, "transport_socket_active", lambda: False)
    monkeypatch.setattr(indy_reads, "INDY_OPERATOR_RESPONSE_OUTBOX", tmp_path / "outbox.jsonl", raising=False)
    monkeypatch.setattr(indy_reads, "mark_conduit_dialogue_done", lambda row, response_id, reply_text: {"ok": False, "error": "unit_no_db"}, raising=False)

    result = indy_reads.record_conduit_dialogue_response(
        {"id": "dialogue-1", "event_id": "$evt", "raw_text": "raw", "clean_text": "clean", "extracted_entities": {}},
        "terminal reply",
        {"slow_lane": {"ingestion_batch_size": 7}},
    )

    assert result["ok"] is False
    assert result["decision"] == "comment"
    assert result["score"] == 100
    assert result["operator_response_queued"] is True
    assert result["db_api_status"] == "db_api_unavailable_fallback"
    assert result["outbound_matrix_send_performed"] is False
    assert result["direct_network_send_performed"] is False
    record = calls["record"]
    assert record["decision"] == "comment"
    assert record["notes"] == "terminal reply"
    assert record["extra"]["response_kind"] == "terminal_conduit_response"
    assert record["extra"]["outbound_matrix_send_performed"] is False
    assert record["batch_size"] == 7


def test_queue_operator_chat_response_writes_quiet_outbox_with_pid_ram_guard(tmp_path: Path, monkeypatch) -> None:
    outbox = tmp_path / "indy_operator_responses.jsonl"
    monkeypatch.setattr(indy_reads, "INDY_OPERATOR_RESPONSE_OUTBOX", outbox, raising=False)
    monkeypatch.setattr(
        indy_reads,
        "hardware_telemetry",
        lambda: {
            "cpu_count": 8,
            "rss_bytes": 12_345,
            "memory_available_bytes": 987_654,
            "memory_percent": 42.0,
        },
    )

    result = indy_reads.queue_operator_chat_response(
        {
            "id": "dialogue-1",
            "event_id": "$evt",
            "room_id": "!room",
            "sender_id": "@operator:local",
            "receipt_id": "matrix_conduit:abc",
        },
        "Indy says hello",
    )

    assert result["ok"] is True
    assert result["operator_delivery_status"] == "QUEUED_FOR_CHAT_SURFACE"
    assert result["outbound_matrix_send_performed"] is False
    assert result["direct_network_send_performed"] is False
    assert outbox.exists()
    packet = json.loads(outbox.read_text(encoding="utf-8").strip())
    assert packet["schema"] == "lucidota.indy_reads.operator_chat_response.v1"
    assert packet["target_path"] == "active_operator_chat_surface"
    assert packet["body"] == "Indy says hello"
    assert packet["db_api_status"] == "db_api_unavailable_fallback"
    assert packet["db_identity"] == {}
    assert packet["dialogue_row"]["event_id"] == "$evt"
    assert packet["pid_ram_guard"]["heavy_model_launch_performed"] is False
    assert packet["pid_ram_guard"]["rss_bytes"] == 12_345
    assert packet["pid_ram_guard"]["memory_available_bytes"] == 987_654


def test_record_conduit_dialogue_response_queues_outbound_and_marks_row_done(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}
    outbox = tmp_path / "indy_operator_responses.jsonl"

    def fake_record(**kwargs):
        calls["record"] = kwargs

    def fake_tune(*args, **kwargs):
        calls["tune"] = {"args": args, "kwargs": kwargs}
        return {}

    def fake_mark(row, response_id, reply_text):
        calls["mark"] = {"row": row, "response_id": response_id, "reply_text": reply_text}
        return {"ok": True, "updated_rows": 1, "processed_status": "done", "response_id": response_id}

    monkeypatch.setattr(indy_reads, "record_indy_judgment", fake_record)
    monkeypatch.setattr(indy_reads, "tune_and_record_heartbeat", fake_tune)
    monkeypatch.setattr(indy_reads, "transport_socket_active", lambda: False)
    monkeypatch.setattr(indy_reads, "INDY_OPERATOR_RESPONSE_OUTBOX", outbox, raising=False)
    monkeypatch.setattr(indy_reads, "mark_conduit_dialogue_done", fake_mark, raising=False)

    row = {"id": "dialogue-1", "event_id": "$evt", "raw_text": "raw", "clean_text": "clean", "extracted_entities": {}}
    result = indy_reads.record_conduit_dialogue_response(row, "terminal reply", {"slow_lane": {"ingestion_batch_size": 7}})

    assert result["ok"] is True
    assert result["operator_response_queued"] is True
    assert result["operator_delivery_status"] == "QUEUED_FOR_CHAT_SURFACE"
    assert result["db_api_status"] == "ok"
    assert result["processed_status_update"] == {
        "ok": True,
        "updated_rows": 1,
        "processed_status": "done",
        "response_id": result["response_id"],
    }
    assert result["outbound_matrix_send_performed"] is False
    assert result["direct_network_send_performed"] is False
    assert Path(result["operator_response_outbox"]).exists()
    assert calls["mark"]["row"] == row
    assert calls["mark"]["response_id"] == result["response_id"]
    assert calls["mark"]["reply_text"] == "terminal reply"
    packet = json.loads(outbox.read_text(encoding="utf-8").strip())
    assert packet["db_api_status"] == "ok"
    assert packet["db_identity"]["response_id"] == result["response_id"]


def test_mark_conduit_dialogue_done_persists_response_identity(monkeypatch) -> None:
    class _MarkCursor:
        def __init__(self):
            self.executed: list[tuple[str, tuple | None]] = []

        def execute(self, sql, params=None):
            self.executed.append((sql, params))

        def fetchall(self):
            return [
                (
                    "dialogue-uuid-1",
                    "done",
                    "indy_response:abc",
                    "indy_response:abc",
                    "QUEUED_FOR_CHAT_SURFACE",
                    "2026-06-04T00:00:00Z",
                    indy_reads.sha_text("terminal reply"),
                )
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _MarkConn:
        def __init__(self):
            self.cursor_obj = _MarkCursor()
            self.committed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    conn = _MarkConn()
    monkeypatch.setattr(indy_reads.psycopg, "connect", lambda dsn: conn)

    result = indy_reads.mark_conduit_dialogue_done(
        {"id": "00000000-0000-0000-0000-000000000001", "event_id": "$evt"},
        "indy_response:abc",
        "terminal reply",
    )

    sql, params = conn.cursor_obj.executed[0]
    assert result["ok"] is True
    assert result["response_id"] == "indy_response:abc"
    assert result["response_delivery_status"] == "QUEUED_FOR_CHAT_SURFACE"
    assert result["response_body_sha256"] == indy_reads.sha_text("terminal reply")
    assert conn.committed is True
    assert "last_response_id" in sql
    assert "last_response_body" in sql
    assert "last_response_body_sha256" in sql
    assert "response_queued_at" in sql
    assert "response_delivery_status" in sql
    assert "RETURNING id::text, processed_status, receipt_id" in sql
    assert params[0] == "indy_response:abc"
    assert params[1] == "indy_response:abc"
    assert params[2] == "terminal reply"


def test_mark_conduit_dialogue_done_rejects_zero_row_identity(monkeypatch) -> None:
    class _ZeroCursor:
        def execute(self, sql, params=None):
            self.sql = sql
            self.params = params

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _ZeroConn:
        def __init__(self):
            self.cursor_obj = _ZeroCursor()
            self.committed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    conn = _ZeroConn()
    monkeypatch.setattr(indy_reads.psycopg, "connect", lambda dsn: conn)

    result = indy_reads.mark_conduit_dialogue_done(
        {"id": "00000000-0000-0000-0000-000000000001"},
        "indy_response:missing",
        "reply",
    )

    assert result["ok"] is False
    assert result["updated_rows"] == 0
    assert result["error"] == "dialogue_row_not_found"
    assert result["response_id"] == "indy_response:missing"
    assert conn.committed is True


def test_process_queued_conduit_once_reads_first_row_and_writes_receipt(tmp_path: Path, monkeypatch) -> None:
    row = {
        "id": "dialogue-1",
        "event_id": "$evt",
        "sender_id": "@operator:local",
        "clean_text": "/indy hello",
        "raw_text": "/indy hello",
        "extracted_entities": {"slash_commands": ["indy"]},
    }
    calls: dict[str, object] = {}
    monkeypatch.setattr(indy_reads, "load_queued_conduit_dialogue", lambda limit=5: [row])
    monkeypatch.setattr(indy_reads, "pid_ram_guard", lambda: {"pid_check_performed": True, "heavy_model_launch_performed": False})

    def fake_record(selected_row, reply_text, st):
        calls["record"] = {"row": selected_row, "reply_text": reply_text, "state": st}
        return {
            "ok": True,
            "response_id": "indy_response:test",
            "operator_response_queued": True,
            "operator_delivery_status": "QUEUED_FOR_CHAT_SURFACE",
            "operator_response_outbox": str(tmp_path / "outbox.jsonl"),
            "processed_status_update": {"ok": True, "processed_status": "done", "updated_rows": 1},
            "outbound_matrix_send_performed": False,
            "direct_network_send_performed": False,
        }

    monkeypatch.setattr(indy_reads, "record_conduit_dialogue_response", fake_record)

    result = indy_reads.process_queued_conduit_once(
        {"slow_lane": {"ingestion_batch_size": 1}},
        receipt_dir=tmp_path,
    )

    assert result["ok"] is True
    assert result["status"] == "RESPONDED"
    assert result["row"]["event_id"] == "$evt"
    assert result["response"]["operator_delivery_status"] == "QUEUED_FOR_CHAT_SURFACE"
    assert result["pid_ram_guard"]["heavy_model_launch_performed"] is False
    assert Path(result["receipt_path"]).exists()
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["schema"] == "lucidota.indy_reads.online_once_receipt.v1"
    assert receipt["model_calls_performed"] is False
    assert receipt["heavy_model_launch_performed"] is False
    assert calls["record"]["row"] == row
    assert calls["record"]["reply_text"].startswith("Indy_READs saw queued chat")


def test_parse_conduit_response_command_selects_queued_row():
    rows = [{"id": "one"}, {"id": "two"}]

    row, reply, error = indy_reads.parse_conduit_response_command("respond 2 hello there", rows)

    assert row == {"id": "two"}
    assert reply == "hello there"
    assert error == ""
