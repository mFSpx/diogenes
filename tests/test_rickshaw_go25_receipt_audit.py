import json
from pathlib import Path

from scripts.rickshaw_go25_receipt_audit import audit_stdout_files, extract_go25_payload


def test_extract_go25_payload_reports_malformed_json():
    text = '{"schema":"lucidota.model_invocation.groq_chat.v1"}\n{"schema":"lucidota.go25.staging_batch.v1","packets":[{"x":1}]\n'
    payload, error = extract_go25_payload(text)
    assert payload is None
    assert error and error.startswith("json_parse_failed:")


def test_audit_stdout_files_detects_duplicate_component_cursor(tmp_path: Path):
    valid = {
        "schema": "lucidota.go25.staging_batch.v1",
        "packets": [
            {
                "source_id": "component_uuid:aaa",
                "parser_name": "groq_rickshaw_go25_extractor.v1",
                "proposed_term": "ENTITY",
                "raw_anchor": "A",
                "claim": "A claim",
                "confidence_bps": 50,
                "status": "pending",
                "proposed_item": {
                    "term": "ENTITY",
                    "status": "staged",
                    "label": "A",
                    "payload": {"source_component_uuid": "aaa"},
                },
                "proposed_edges": [],
            }
        ],
    }
    files = []
    for batch in (42, 43):
        p = tmp_path / f"groq_go25_batch_{batch:03d}_x.stdout.jsonl"
        p.write_text(json.dumps({"schema": "receipt"}) + "\n" + json.dumps(valid) + "\n")
        files.append(p)

    report = audit_stdout_files(files, validate=False)

    assert report["counts"]["stdout_files"] == 2
    assert report["counts"]["valid_json"] == 2
    assert report["counts"]["duplicate_component_batches"] == 1
    assert report["duplicate_component_batches"][0]["component_uuid"] == "aaa"


def test_extract_go25_payload_accepts_fenced_json_block():
    payload = {"schema": "lucidota.go25.staging_batch.v1", "packets": []}
    text = "prefix\n```json\n" + json.dumps(payload) + "\n```\nRECEIPT_PATH=x"
    parsed, error = extract_go25_payload(text)
    assert error is None
    assert parsed == payload


def test_extract_go25_payload_accepts_pretty_fenced_json_block():
    payload = {"schema": "lucidota.go25.staging_batch.v1", "packets": []}
    text = "prefix\n```json\n" + json.dumps(payload, indent=2) + "\n```\nRECEIPT_PATH=x"
    parsed, error = extract_go25_payload(text)
    assert error is None
    assert parsed == payload


def test_next_selected_batch_041_is_distinct_from_previous_staged_batch_042():
    batch_041 = json.loads(
        Path("05_OUTPUTS/rickshaw_reingest/groq_go25_batch_041_20260528T012521Z.schema_repaired.json").read_text()
    )
    batch_042 = json.loads(
        Path("05_OUTPUTS/rickshaw_reingest/groq_go25_batch_042_repair_packet_20260528T0448Z.json").read_text()
    )
    comp_041 = batch_041["packets"][0]["proposed_item"]["payload"]["source_component_uuid"]
    comp_042 = batch_042["packets"][0]["proposed_item"]["payload"]["source_component_uuid"]
    assert comp_041 != comp_042
