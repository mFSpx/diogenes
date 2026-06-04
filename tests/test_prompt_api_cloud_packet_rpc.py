from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from uuid import uuid4

import pytest


def test_cloud_packet_rpc_returns_bounded_json_from_postgrest() -> None:
    psycopg = pytest.importorskip("psycopg")
    database_url = "postgresql:///lucidota_state"
    base_url = "http://127.0.0.1:3000"
    event_id = hashlib.sha256(b"pytest-cloud-packet-event").hexdigest()
    verbatim_hash = hashlib.sha256(b"pytest-cloud-packet-text").hexdigest()
    raw_ref = "inline://pytest-cloud-packet"
    work_order_uuid = None
    unique_suffix = uuid4().hex

    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
                VALUES (%s, %s, 'sha256', 'pytest', 'worker', 24, 24, 'text/plain', 'inline', '{}'::jsonb)
                ON CONFLICT (raw_ref) DO UPDATE SET detail = EXCLUDED.detail
                RETURNING raw_artifact_uuid::text
                """,
                (raw_ref, hashlib.sha256(b"pytest-cloud-packet-raw").hexdigest()),
            )
            raw_artifact_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
                VALUES (%s, now(), 'pytest', 'worker', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, '{}'::jsonb, NULL, '{}'::jsonb)
                ON CONFLICT (event_id) DO UPDATE SET detail = EXCLUDED.detail
                """,
                (event_id, raw_ref, raw_artifact_uuid, verbatim_hash, "pytest cloud packet event"),
            )
            cur.execute(
                """
                INSERT INTO lucidota_control.route_decision(event_id, lane, route_key, deterministic_rule, treelite_gate, model_fallback, cost, expected_gain, confidence, graph_write_mode, detail)
                VALUES (%s, 'slow', 'pytest', '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, 0, 0.5, 'staged_only', '{}'::jsonb)
                RETURNING decision_uuid::text
                """,
                (event_id,),
            )
            decision_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.work_order(event_id, decision_uuid, lane, work_kind, status, payload, idempotency_key)
                VALUES (%s, %s::uuid, 'slow', 'pytest_cloud_packet', 'queued', %s::jsonb, %s)
                RETURNING work_order_uuid::text
                """,
                (event_id, decision_uuid, json.dumps({"task_type": "repair", "target_model": "codex", "summary": "bounded prompt packet", "next_action": "continue local execution"}), f"pytest-cloud-packet:{event_id}:{unique_suffix}"),
            )
            work_order_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_learning.bytewax_compact_window(
                    work_order_uuid, work_order_id, source, topic, object_type, window_kind, window_start_at, window_end_at,
                    event_count, dropped_raw_bodies, summary, features, scores, needs_cloud_reasoning, event_ids, source_hashes, receipt_refs, detail
                ) VALUES (
                    %s::uuid, %s, 'pytest', 'repair', 'Receipt', 'tumbling', now(), now(),
                    1, 1, 'pytest compact window', %s::jsonb, %s::jsonb, true, %s::jsonb, %s::jsonb, %s::jsonb, '{}'::jsonb
                )
                """,
                (
                    work_order_uuid,
                    work_order_uuid,
                    json.dumps({"event_count": 1, "unique_event_ids": 1, "unique_source_hashes": 1, "unique_receipt_refs": 1, "time_span_seconds": 0}),
                    json.dumps({"local_score": 0.9, "treelite_score": 0.25, "treelite_lane": "slow"}),
                    json.dumps([event_id]),
                    json.dumps(["pytest-source-hash"]),
                    json.dumps(["pytest-receipt"]),
                ),
            )
            conn.commit()
    except Exception as exc:
        pytest.skip(f"database unavailable for RPC setup: {type(exc).__name__}: {exc}")

    req = urllib.request.Request(
        f"{base_url}/rpc/cloud_packet",
        data=json.dumps(
            {
                "work_order_id": work_order_uuid,
                "max_chars": 512,
                "max_items": 4,
                "task_type": "repair",
                "target_model": "codex",
                "include_raw_bodies": False,
            }
        ).encode("utf-8"),
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    assert payload["contract_name"] == "prompt_api.cloud_packet.v1"
    assert payload["work_order_id"] == work_order_uuid
    assert payload["rules"]["max_chars"] <= 512
    assert payload["rules"]["max_items"] <= 4
    assert payload["summary"]["window_count"] >= 1
    assert payload["selected_evidence_refs"]
    assert payload["event_ids"]
    assert payload["source_hashes"]
    assert payload["scores"]["local_score"] >= 0
    assert payload["scores"]["treelite_score"] >= 0
    assert payload["needs_cloud_reasoning"] is True
    assert "raw_bodies" in payload
    assert payload["raw_bodies"] == []
