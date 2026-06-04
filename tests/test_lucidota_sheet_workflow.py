import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

SCHEMA = Path("06_SCHEMA/052_lucidota_sheet_workflow_layer.sql")
SCRIPT = Path("scripts/lucidota_sheet_workflow_smoke.py")
WORKFLOW_MANIFEST = Path("04_RUNTIME/lucidota_workflow_registry.json")
REQUIRED_ORDER = ["ingest", "evidence_ingest", "graph_ops", "documents_forms", "network_analysis"]


def test_sheet_workflow_spine_sql_declares_schemas_tables_views_and_receipts():
    text = SCHEMA.read_text(encoding="utf-8")
    for needle in [
        "CREATE SCHEMA IF NOT EXISTS lucidota_sheet;",
        "CREATE SCHEMA IF NOT EXISTS lucidota_scratch;",
        "CREATE SCHEMA IF NOT EXISTS lucidota_projection;",
        "CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_workflow_route",
        "CREATE TABLE IF NOT EXISTS lucidota_sheet.sheet_workflow_receipt",
        "CREATE UNLOGGED TABLE IF NOT EXISTS lucidota_scratch.sheet_workflow_route_scratch",
        "CREATE OR REPLACE VIEW lucidota_sheet.sheet_workflow_head",
        "CREATE OR REPLACE VIEW lucidota_projection.sheet_workflow_route_sheet",
        "CREATE MATERIALIZED VIEW IF NOT EXISTS lucidota_projection.workflow_domain_pressure_sheet",
        "CREATE OR REPLACE FUNCTION lucidota_sheet.record_sheet_workflow_receipt",
    ]:
        assert needle in text


def validate_receipt(receipt: dict[str, object]) -> None:
    assert receipt["schema"] == "lucidota.sheet_workflow_smoke_receipt.v1"
    assert receipt["execution"] == "dry_run"
    assert receipt["db_connected"] is False
    assert receipt["status"] in {"PASS", "FAIL"}
    assert receipt["route_order"] == REQUIRED_ORDER
    assert len(receipt["required_domains"]) == len(REQUIRED_ORDER)
    for route in receipt["routes"]:
        assert route["logical_domain"] in REQUIRED_ORDER
        assert route["target"]
        assert route["query_hash"]
        assert route["query_sql"]
        assert route["query_has_limit"] is True or "REFRESH MATERIALIZED VIEW" in route["query_sql"].upper()
        assert "SELECT *" not in route["query_sql"].upper()


def test_sheet_workflow_smoke_runs_dry_run_without_db_and_hashes_receipt(tmp_path):
    receipt_path = tmp_path / "sheet_workflow_smoke.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workflow",
            str(WORKFLOW_MANIFEST),
            "--receipt",
            str(receipt_path),
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["execution"] == "dry_run"
    assert payload["db_connected"] is False
    assert payload["route_order"] == REQUIRED_ORDER

    validate_receipt(payload)

    assert receipt_path.exists()
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert saved == payload

    # output hash must be valid and match deterministic schema
    output_fields = {k: payload[k] for k in payload if k != "output_hash"}
    expected = sha256(json.dumps(output_fields, sort_keys=True).encode("utf-8")).hexdigest()
    assert payload["output_hash"] == expected


def test_sheet_workflow_smoke_contains_expected_domain_targets_from_registry():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--workflow", str(WORKFLOW_MANIFEST), "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    present = [route["logical_domain"] for route in payload["routes"]]
    assert present == REQUIRED_ORDER
    route_targets = {route["logical_domain"]: route["target"] for route in payload["routes"]}
    assert route_targets == {
        "ingest": "workflow.korpus_ingest.route",
        "evidence_ingest": "workflow.evidence_ingest.capture",
        "graph_ops": "workflow.graph_ops.materialize",
        "documents_forms": "workflow.documents_forms.packetize",
        "network_analysis": "workflow.network_analysis.centrality",
    }
