from __future__ import annotations

import os
import uuid

import pytest

from scripts import indy_daemon


def _db_url() -> str:
    return os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def test_indy_daemon_once_processes_seeded_queued_row_and_writes_response() -> None:
    psycopg = pytest.importorskip("psycopg")
    dsn = _db_url()

    event_id = f"$pytest-indy-daemon-{uuid.uuid4().hex}"
    sender_id = "@pytest:local"
    room_id = "!pytest:local"
    receipt_id = f"pytest:{uuid.uuid4().hex}"
    seeded_row_id = None

    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ironclaw.waking_dialogue_stream
                  (comms_channel, sender_id, room_id, event_id, raw_text, clean_text, extracted_entities, processed_status, receipt_id, received_at)
                VALUES
                  ('matrix', %s, %s, %s, %s, %s, '{}'::jsonb, 'queued', %s, '1970-01-01T00:00:00Z')
                ON CONFLICT (comms_channel, event_id) DO UPDATE SET
                  raw_text = EXCLUDED.raw_text,
                  clean_text = EXCLUDED.clean_text,
                  extracted_entities = EXCLUDED.extracted_entities,
                  processed_status = EXCLUDED.processed_status,
                  receipt_id = EXCLUDED.receipt_id,
                  received_at = EXCLUDED.received_at,
                  updated_at = now()
                RETURNING id::text;
                """,
                (sender_id, room_id, event_id, "/indy pytest daemon once", "/indy pytest daemon once", receipt_id),
            )
            row = cur.fetchone()
            seeded_row_id = row[0] if row else None
            conn.commit()

        result = indy_daemon.run_once(base_url="http://127.0.0.1:3000", limit=1, max_items=1)
        assert result["poll"]["visible_route"] == "/indy_queue"
        assert result["poll"]["row_count"] >= 1
        assert result["responded"] is True

        with psycopg.connect(dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT processed_status, last_response_id, response_queued_at, response_delivery_status
                FROM ironclaw.waking_dialogue_stream
                WHERE comms_channel = 'matrix' AND event_id = %s;
                """,
                (event_id,),
            )
            row = cur.fetchone()

        assert row is not None
        assert row[0] == "done"
        assert row[1]
        assert row[2] is not None
        assert row[3] == "QUEUED_FOR_CHAT_SURFACE"
        assert "BOOKS" not in result["respond_stdout"]
    finally:
        if seeded_row_id:
            try:
                with psycopg.connect(dsn, connect_timeout=3) as conn, conn.cursor() as cur:
                    cur.execute("DELETE FROM ironclaw.waking_dialogue_stream WHERE id = %s::uuid", (seeded_row_id,))
                    conn.commit()
            except Exception:
                pass
