import json
from pathlib import Path


MANIFEST = Path("04_RUNTIME/korpus_sheet_first_ingest_workflow.json")


def test_korpus_sheet_first_ingest_workflow_is_saved_and_routes_before_llms():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "lucidota.korpus.sheet_first_ingest_workflow.v1"
    assert data["status"] == "STAGED_WORKFLOW_READY_TO_RUN"
    assert data["routing_order"][:4] == [
        "inventory_sheet",
        "quality_sheet",
        "deterministic_extract",
        "bounded_algos",
    ]
    assert data["routing_order"][-1] == "llm_last_resort"
    assert data["tokio_pubsub"]["executor"] == "tokio_bounded_pubsub"
    assert data["absurd"]["lane"] == "slowlane_only_after_sheet_algo_gap"
    assert data["body_policy"] == "refs_not_bodies"
    assert data["workflows"]["ingest"]["script_candidates"]
    assert "scripts/lucidota_ingestion_quality_audit.py" in data["workflows"]["quality_gate"]["script_candidates"]
    assert data["receipts"]["required"] is True
    assert data["limits"]["max_file_read_bytes"] > 0
    assert data["postgrest"]["sheet_api_probe"] == "05_OUTPUTS/runtime/postgrest_sheet_api_probe_latest.json"
