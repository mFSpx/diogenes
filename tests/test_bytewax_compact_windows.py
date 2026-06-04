from __future__ import annotations

from scripts.bytewax_abductive_blender import BlenderEvent, build_compact_windows


def test_build_compact_windows_groups_by_work_order_topic_and_object_type() -> None:
    events = [
        BlenderEvent(
            source="workflow_event",
            source_ref="evt-1",
            event_time="2026-06-04T00:00:00Z",
            text_surface="queued work order",
            payload={
                "work_order_id": "wo-1",
                "topic": "ingest",
                "object_type": "Receipt",
                "source_hash": "hash-a",
                "receipt_ref": "receipt-a",
                "summary": "first body",
            },
            compressed_activity={"key_count": 2},
        ),
        BlenderEvent(
            source="workflow_event",
            source_ref="evt-2",
            event_time="2026-06-04T00:00:15Z",
            text_surface="queued work order",
            payload={
                "work_order_id": "wo-1",
                "topic": "ingest",
                "object_type": "Receipt",
                "source_hash": "hash-b",
                "receipt_ref": "receipt-b",
                "summary": "second body",
            },
            compressed_activity={"key_count": 1},
        ),
    ]

    windows = build_compact_windows(events, run_id="run-1")

    assert windows
    assert {row["work_order_id"] for row in windows} == {"wo-1"}
    assert {row["source"] for row in windows} == {"workflow_event"}
    assert all(row["dropped_raw_bodies"] >= 1 for row in windows)
    assert all("raw_body" not in row for row in windows)
    assert all(isinstance(row["event_ids"], list) and row["event_ids"] for row in windows)
    assert all(isinstance(row["source_hashes"], list) and row["source_hashes"] for row in windows)
    assert all("local_score" in row["scores"] and "treelite_score" in row["scores"] for row in windows)

