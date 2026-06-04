import json
import subprocess
import sys
from pathlib import Path

SQL = Path("06_SCHEMA/147_lucidota_sheet_layer.sql")
MANIFEST = Path("04_RUNTIME/lucidota_sheet_manifest.json")
CLI = Path("scripts/luci_sheet.py")


def test_sheet_layer_sql_declares_required_schemas_and_sheet_primitives():
    text = SQL.read_text(encoding="utf-8")
    for schema in ["lucidota_sheet", "lucidota_scratch", "lucidota_projection"]:
        assert f"CREATE SCHEMA IF NOT EXISTS {schema}" in text
    assert "GENERATED ALWAYS AS" in text
    assert "CREATE OR REPLACE VIEW lucidota_sheet.active_work" in text
    assert "CREATE MATERIALIZED VIEW IF NOT EXISTS lucidota_projection.case_pressure_sheet" in text
    assert "CREATE UNLOGGED TABLE IF NOT EXISTS lucidota_scratch.route_score_scratch" in text
    assert "CREATE UNLOGGED TABLE IF NOT EXISTS lucidota_scratch.runpod_chunk_embedding_stage" in text
    assert "embedding_json jsonb NOT NULL,\n  error text" in text
    assert "ALTER COLUMN row_json SET DEFAULT '{}'::jsonb" in text
    assert "CREATE OR REPLACE VIEW lucidota_projection.runpod_chunk_embedding_sheet" in text
    assert "REFRESH MATERIALIZED VIEW" in text
    assert "COPY (" in text
    assert "lucidota_sheet.record_refresh_receipt" in text


def test_manifest_registers_sheet_tasks_before_algorithm_escalation():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "lucidota.sheet_manifest.v1"
    assert data["routing_order"][:6] == [
        "generated_column",
        "live_view",
        "materialized_projection",
        "sql_aggregate",
        "duckdb_file_sheet",
        "algorithm_escalation",
    ]
    assert data["routing_order"][-1] == "model_last_resort"
    classes = {c["id"] for c in data["sheet_task_classes"]}
    assert classes >= {
        "FILTER_SHEET",
        "STATUS_SHEET",
        "PIVOT_SHEET",
        "SCORE_SHEET",
        "DIFF_SHEET",
        "REFRESH_SHEET",
        "EXPORT_SHEET",
        "IMPORT_SHEET",
        "PROMOTION_SHEET",
        "DEADLETTER_SHEET",
    }
    for sheet in data["sheets"]:
        assert sheet["max_rows"] > 0
        assert sheet["receipt_required"] is True
        assert "SELECT *" not in sheet.get("query", "").upper()
    ids = {sheet["id"] for sheet in data["sheets"]}
    assert "runpod_chunk_embedding_sheet" in ids


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), "--manifest", str(MANIFEST), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_luci_sheet_cli_lists_and_shows_registered_sheets():
    listed = run_cli("list", "--json")
    assert listed.returncode == 0, listed.stderr
    rows = json.loads(listed.stdout)
    ids = {row["id"] for row in rows["sheets"]}
    assert {"active_work", "next_work_batch", "case_pressure_sheet"} <= ids

    shown = run_cli("show", "active_work", "--json")
    assert shown.returncode == 0, shown.stderr
    sheet = json.loads(shown.stdout)
    assert sheet["id"] == "active_work"
    assert sheet["kind"] == "VIEW"
    assert sheet["max_rows"] <= 1000

    current = run_cli("current", "--json")
    assert current.returncode == 0, current.stderr
    current_payload = json.loads(current.stdout)
    assert current_payload["current_route"] == "/sheet_current"
    assert current_payload["current_object"] == "lucidota_sheet.sheet_current"


def test_luci_sheet_cli_explains_refresh_export_and_rejects_unknown():
    explained = run_cli("explain", "next_work_batch", "--json")
    assert explained.returncode == 0, explained.stderr
    plan = json.loads(explained.stdout)
    assert plan["sheet"] == "next_work_batch"
    assert plan["execution"] == "dry_run"
    assert "LIMIT" in plan["query"].upper()

    refresh = run_cli("refresh", "case_pressure_sheet", "--json")
    assert refresh.returncode == 0, refresh.stderr
    receipt = json.loads(refresh.stdout)
    assert receipt["operation"] == "refresh_projection"
    assert receipt["receipt_required"] is True
    assert receipt["sql"].startswith("REFRESH MATERIALIZED VIEW")

    export = run_cli("export", "next_work_batch", "--format", "csv", "--json")
    assert export.returncode == 0, export.stderr
    export_plan = json.loads(export.stdout)
    assert export_plan["operation"] == "export_sheet"
    assert "COPY (" in export_plan["sql"]

    bad = run_cli("show", "nope", "--json")
    assert bad.returncode == 2
    assert json.loads(bad.stdout)["error"] == "unknown_sheet"
