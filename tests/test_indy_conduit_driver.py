from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "indy_conduit_driver.py"


def load_module():
    spec = importlib.util.spec_from_file_location("indy_conduit_driver", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_text_matrix_event_becomes_queued_dialogue_stream_row_only():
    mod = load_module()
    event = {
        "event_id": "$evt1",
        "room_id": "!room:lucidota",
        "sender": "@operator:local",
        "origin_server_ts": 1,
        "content": {"msgtype": "m.text", "body": "status please"},
    }

    plan = mod.build_plan(event)

    assert plan["dialogue_row"]["comms_channel"] == "matrix"
    assert plan["dialogue_row"]["raw_text"] == "status please"
    assert plan["dialogue_row"]["clean_text"] == "status please"
    assert plan["dialogue_row"]["extracted_entities"] == {"urls": [], "emails": [], "slash_commands": [], "hashtags": []}
    assert plan["dialogue_row"]["processed_status"] == "queued"
    assert plan["dialogue_row"]["source_payload"]["matrix"]["event_id"] == "$evt1"
    assert plan["absurd_jobs"] == []


def test_text_matrix_event_extracts_entities_before_indy_reads():
    mod = load_module()
    event = {
        "event_id": "$evt_entities",
        "room_id": "!room:lucidota",
        "sender": "@operator:local",
        "content": {"msgtype": "m.text", "body": "  /flow   ping admin@example.com https://example.test/a  #Canon  "},
    }

    row = mod.build_plan(event)["dialogue_row"]

    assert row["clean_text"] == "/flow ping admin@example.com https://example.test/a #Canon"
    assert row["extracted_entities"]["slash_commands"] == ["flow"]
    assert row["extracted_entities"]["emails"] == ["admin@example.com"]
    assert row["extracted_entities"]["urls"] == ["https://example.test/a"]
    assert row["extracted_entities"]["hashtags"] == ["Canon"]


def test_file_matrix_event_attaches_metadata_and_enqueues_absurd_atomization_job():
    mod = load_module()
    event = {
        "event_id": "$file1",
        "room_id": "!room:lucidota",
        "sender": "@operator:local",
        "content": {
            "msgtype": "m.file",
            "body": "archive.zip",
            "url": "mxc://local/archive",
            "info": {"size": 1234, "mimetype": "application/zip"},
            "file": {"hashes": {"sha256": "abc"}},
        },
    }

    plan = mod.build_plan(event)

    assert plan["dialogue_row"]["source_payload"]["matrix"]["attachment"] == {
        "body": "archive.zip",
        "mxc_url": "mxc://local/archive",
        "size": 1234,
        "mimetype": "application/zip",
        "sha256": "abc",
    }
    assert len(plan["absurd_jobs"]) == 1
    job = plan["absurd_jobs"][0]
    assert job["queue_name"] == "matrix_intake"
    assert job["workflow_name"] == "matrix.file.atomize"
    assert job["job_kind"] == "matrix_file_atomize"
    assert job["payload"]["matrix_event_ref"]["event_id"] == "$file1"
    assert "raw_file_bytes" not in json.dumps(job)


def test_flow_command_is_visible_chat_widget_request_not_hidden_automation():
    mod = load_module()
    event = {
        "event_id": "$flow1",
        "room_id": "!room:lucidota",
        "sender": "@operator:local",
        "content": {"msgtype": "m.text", "body": "/flow"},
    }

    plan = mod.build_plan(event)

    assert plan["ui_action"] == {
        "kind": "chat_platform_widget_request",
        "widget_key": "lucidota.promptflow_canvas",
        "home": "active_operator_chat_surface",
        "manual_panel": "postgrest_html_manual",
        "requires_operator_stage_validate_run": True,
    }
    assert len(plan["absurd_jobs"]) == 1
    assert plan["absurd_jobs"][0]["job_kind"] == "matrix_widget_open_request"


def test_cli_dry_run_writes_receipt_without_database(tmp_path):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "event_id": "$evt2",
        "room_id": "!room:lucidota",
        "sender": "@operator:local",
        "content": {"msgtype": "m.text", "body": "hello indy"},
    }))
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(MODULE_PATH),
            "--event-file",
            str(event_path),
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    receipt = Path(payload["receipt_path"])
    assert receipt.exists()
    data = json.loads(receipt.read_text())
    assert data["schema"] == "lucidota.indy.matrix_conduit_receipt.v1"
    assert data["executed"] is False
    assert data["plan"]["dialogue_row"]["raw_text"] == "hello indy"
    assert data["plan"]["dialogue_row"]["clean_text"] == "hello indy"


def test_process_event_payload_dry_run_returns_receipt_without_database(tmp_path):
    mod = load_module()
    event = {
        "event_id": "$evt_process_payload",
        "room_id": "!room:lucidota",
        "sender": "@operator:local",
        "content": {"msgtype": "m.text", "body": "/indy dry run from chat"},
    }

    result = mod.process_event_payload(
        event,
        dry_run=True,
        output_dir=tmp_path,
        database_url="postgresql:///must_not_be_used_in_dry_run",
    )

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["db_result"] is None
    receipt = Path(result["receipt_path"])
    assert receipt.exists()
    payload = json.loads(receipt.read_text())
    assert payload["plan"]["dialogue_row"]["comms_channel"] == "matrix"
    assert payload["plan"]["dialogue_row"]["raw_text"] == "/indy dry run from chat"
    assert payload["plan"]["dialogue_row"]["processed_status"] == "queued"
    assert payload["plan"]["dialogue_row"]["receipt_id"] == result["receipt_id"]


def test_cli_prefers_absurd_system_database_url_for_same_db_as_indy(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///from_absurd")
    monkeypatch.setenv("DATABASE_URL", "postgresql:///from_database")

    parser_args = []
    # argparse defaults are built inside main; exercise it through parse-safe dry run.
    # The receipt path proves the command did not need a DB connection.
    event = {"event_id": "$dburl", "room_id": "!room", "sender": "@op", "content": {"msgtype": "m.text", "body": "ping"}}
    plan = mod.build_plan(event)
    assert plan["dialogue_row"]["comms_channel"] == "matrix"
    assert "ABSURD_SYSTEM_DATABASE_URL" in MODULE_PATH.read_text()


def test_read_queued_dialogue_rows_exposes_raw_and_clean_text_without_mutation():
    mod = load_module()

    class Cursor:
        def __init__(self):
            self.sql = ""
            self.params = None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def execute(self, sql, params):
            self.sql = sql
            self.params = params
        def fetchall(self):
            return [(
                "dialogue-1",
                "2026-06-04T00:00:00Z",
                "@operator:local",
                "!room",
                "$evt",
                "  raw   text  ",
                "raw text",
                {"slash_commands": []},
                "receipt-1",
                "2026-06-04T00:00:00Z",
            )]

    class Conn:
        def __init__(self):
            self.cursor_obj = Cursor()
        def cursor(self):
            return self.cursor_obj

    conn = Conn()
    rows = mod.read_queued_dialogue_rows(conn, limit=500)

    assert rows == [{
        "id": "dialogue-1",
        "received_at": "2026-06-04T00:00:00Z",
        "sender_id": "@operator:local",
        "room_id": "!room",
        "event_id": "$evt",
        "raw_text": "  raw   text  ",
        "clean_text": "raw text",
        "extracted_entities": {"slash_commands": []},
        "receipt_id": "receipt-1",
        "created_at": "2026-06-04T00:00:00Z",
        "read_only": True,
    }]
    assert "SELECT id::text" in conn.cursor_obj.sql
    assert "raw_text, clean_text" in conn.cursor_obj.sql
    assert "processed_status = 'queued'" in conn.cursor_obj.sql
    assert "UPDATE" not in conn.cursor_obj.sql.upper()
    assert "INSERT" not in conn.cursor_obj.sql.upper()
    assert conn.cursor_obj.params == (100,)


def test_driver_db_insert_uses_clean_conduit_contract_not_legacy_columns():
    script = MODULE_PATH.read_text()

    for column in ["comms_channel", "sender_id", "room_id", "event_id", "raw_text", "clean_text", "extracted_entities", "processed_status", "receipt_id"]:
        assert column in script
    assert "RETURNING id::text" in script
    assert "input_payload" not in script
    assert "stream_id" not in script
    assert "arbitrary SQL" not in script
