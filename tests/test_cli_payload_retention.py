from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from psycopg.rows import dict_row


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cli_payload_retention.py"
LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_cli_payload_retention_archives_cli_receipt_tails_and_exposes_status(tmp_path):
    psycopg = pytest.importorskip("psycopg")
    receipt_uuid = uuid4()
    command_line = "fake-cli --hello"
    stdout_tail = "stdout line 1\nstdout line 2"
    stderr_tail = "stderr line 1"
    try:
        with psycopg.connect("postgresql:///lucidota_state") as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.cli_process_receipt (
                    receipt_uuid, received_at, command_line, command_sha256, process_pid,
                    timeout_seconds, restart_count, auth_env_var, auth_prompt_seen,
                    auth_injected, status, exit_code, stdout_tail, stderr_tail,
                    receipt_path, detail
                ) VALUES (
                    %s::uuid, now() - interval '365 days', %s, repeat('a', 64), 42, 1, 0, 'TEST_AUTH_TOKEN', true,
                    true, 'succeeded', 0, %s, %s, 'receipts/cli.json', '{}'::jsonb
                )
                ON CONFLICT (receipt_uuid) DO UPDATE SET
                    stdout_tail = EXCLUDED.stdout_tail,
                    stderr_tail = EXCLUDED.stderr_tail,
                    updated_at = now()
                """,
                (str(receipt_uuid), command_line, stdout_tail, stderr_tail),
            )
            conn.commit()
    except Exception as exc:
        pytest.skip(f"database unavailable for retention setup: {type(exc).__name__}: {exc}")

    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPT),
            "--archive-all",
            "--max-rows",
            "1",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "lucidota.cli_payload_retention.v1"
    assert payload["count"] >= 1

    with psycopg.connect("postgresql:///lucidota_state", row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT stdout_tail, stderr_tail, stdout_archive_ref, stderr_archive_ref, stdout_tail_sha256, stderr_tail_sha256 FROM lucidota_control.cli_process_receipt WHERE receipt_uuid = %s::uuid",
            (str(receipt_uuid),),
        )
        row = cur.fetchone()
        assert row["stdout_tail"] == ""
        assert row["stderr_tail"] == ""
        assert row["stdout_archive_ref"]
        assert row["stderr_archive_ref"]
        assert row["stdout_tail_sha256"]
        assert row["stderr_tail_sha256"]

    import urllib.request

    with urllib.request.urlopen(f"{LIVE_BASE_URL}/payload_archive_status?limit=10", timeout=5) as resp:
        assert resp.status == 200
        rows = json.loads(resp.read().decode("utf-8"))
    assert isinstance(rows, list)
    assert rows
    assert rows[0]["source_table"] == "lucidota_control.cli_process_receipt"
    assert "next_command_refs" in rows[0]
    assert isinstance(rows[0]["next_command_refs"], list) and rows[0]["next_command_refs"]
    assert "manual_current" in rows[0]["next_command_refs"]
    assert "root_orchestrator_current" in rows[0]["next_command_refs"]
    assert "daemon_status" in rows[0]["next_command_refs"]
    assert "cli_process_receipts" in rows[0]["next_command_refs"]
    assert "command_registry" in rows[0]["next_command_refs"]
    assert isinstance(rows[0].get("orchestration"), dict)
    assert rows[0]["orchestration"]["mode"] == "sub_orchestrator"


def test_cli_payload_archive_render_shows_next_refs():
    from scripts.luci_payload_archive import render

    out = render(
        [
            {
                "source_table": "lucidota_control.cli_process_receipt",
                "payload_kind": "stdout_tail",
                "archive_count": 2,
                "archived_bytes": 42,
                "archived_chars": 99,
                "latest_archived_at": "2026-06-04T00:00:00Z",
                "next_command_refs": ["manual_current", "root_orchestrator_current"],
            }
        ]
    )
    assert "next_command_refs=manual_current, root_orchestrator_current" in out
