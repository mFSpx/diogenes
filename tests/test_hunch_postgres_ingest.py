from __future__ import annotations

import json
from pathlib import Path

from scripts.hunch_postgres_ingest import (
    build_graph_stage_packets,
    build_manual_hunch_record,
    build_upsert_rows,
    load_hunch_records,
    write_ingest_observation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_build_manual_hunch_record_labels_new_hunch_without_truth_promotion() -> None:
    record = build_manual_hunch_record(
        text="This is gonna work, this time. Ingest. Porges. No Choke.",
        source="operator_chat",
        root=ROOT,
    )

    assert record["hunch_id"].startswith("OP-")
    assert record["rating"] == "OPEN"
    assert record["evidence_state"] == "operator_fresh_hunch"
    assert record["truth_promotion"] == "blocked_until_evidence_paths_reviewed"
    assert record["source_sha256"]
    assert "Porges" in record["title"]


def test_load_hunch_records_accepts_audit_report_and_manual_hunch() -> None:
    audit_path = ROOT / "04_RUNTIME" / "observation_center" / "hunch_hypertimeline_latest.json"
    records = load_hunch_records(audit_path, manual_hunch="This is gonna work, this time. Ingest. Porges. No Choke.", root=ROOT)

    assert any(r["hunch_id"].startswith("OP-") for r in records)
    assert len(records) >= 1


def test_build_upsert_rows_and_graph_packets_are_stage_only() -> None:
    records = [
        build_manual_hunch_record(
            text="This is gonna work, this time. Ingest. Porges. No Choke.",
            source="operator_chat",
            root=ROOT,
        )
    ]

    rows = build_upsert_rows(records, root=ROOT)
    packets = build_graph_stage_packets(rows)

    assert rows[0]["canonical_graph_writes_performed"] is False
    assert rows[0]["authority_class"] == "operator_hunch_signal_not_truth"
    assert packets[0]["kind"] == "OBJECT"
    assert packets[0]["term"] == "HUNCH"
    assert packets[0]["promotion_state"] == "staged_candidate_only"
    assert json.dumps(packets)


def test_write_ingest_observation_updates_runtime_and_big_board(tmp_path: Path) -> None:
    rows = [
        build_upsert_rows(
            [
                build_manual_hunch_record(
                    text="This is gonna work, this time. Ingest. Porges. No Choke.",
                    source="operator_chat",
                    root=ROOT,
                )
            ],
            root=ROOT,
        )[0]
    ]
    receipt = {
        "schema": "lucidota.hunch_postgres_ingest.receipt.v1",
        "generated_at": "2026-05-27T00:00:00Z",
        "records_seen": 1,
        "records_upserted": 1,
        "graph_candidates_written": 1,
        "manual_hunch_present": True,
        "receipt_path": "05_OUTPUTS/hunch_hypertimeline/example.json",
        "graph_stage_path": "05_OUTPUTS/hunch_hypertimeline/example.jsonl",
        "canonical_graph_writes_performed": False,
    }
    (tmp_path / "05_OUTPUTS").mkdir(parents=True)
    (tmp_path / "05_OUTPUTS" / "big_board.json").write_text(json.dumps({"counters": {}}), encoding="utf-8")

    result = write_ingest_observation(receipt, rows, root=tmp_path)

    runtime = tmp_path / "04_RUNTIME" / "observation_center" / "hunch_postgres_ingest_latest.json"
    big_board = json.loads((tmp_path / "05_OUTPUTS" / "big_board.json").read_text(encoding="utf-8"))
    assert runtime.exists()
    assert result["observation_center_path"] == "04_RUNTIME/observation_center/hunch_postgres_ingest_latest.json"
    assert big_board["observation_center"]["hunch_postgres_ingest"]["records_upserted"] == 1
    assert big_board["counters"]["hunch_postgres_records_upserted"] == 1
