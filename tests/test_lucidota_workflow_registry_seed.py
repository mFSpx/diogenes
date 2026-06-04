import json
import subprocess
from pathlib import Path

REGISTRY = Path("04_RUNTIME/lucidota_workflow_registry.json")
RECEIPT = Path("05_OUTPUTS/runtime/lucidota_workflow_seed_latest.json")


def run_seed(*args):
    return subprocess.run(
        ["python3", "scripts/lucidota_workflow_task_seed.py", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_workflow_registry_covers_more_than_korpus():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    domains = {w["domain"] for w in data["workflows"]}
    assert {
        "korpus_ingest",
        "evidence_ingest",
        "graph_ops",
        "network_analysis",
        "pivot_search",
        "documents_forms",
        "workflow_automation",
        "runpod_artifact_import",
    }.issubset(domains)
    assert data["routing_law"][:3] == ["sheet", "treelite_or_router", "bounded_algo"]
    for workflow in data["workflows"]:
        assert workflow["task_type"] == "SHEET_TASK"
        assert workflow["task_class"] in data["allowed_task_classes"]
        assert workflow["receipt_required"] is True
        assert workflow["body_policy"] == "refs_not_bodies"
        assert workflow["max_rows"] <= data["limits"]["max_db_rows_per_batch"]
        assert "SELECT *" not in workflow["query_sql"].upper()
        assert "LIMIT" in workflow["query_sql"].upper() or workflow["task_class"] == "REFRESH_SHEET"


def test_workflow_seed_dry_run_emits_all_domains_and_small_receipt():
    result = run_seed("--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["execution"] == "dry_run"
    assert payload["tasks_seen"] >= 7
    assert payload["would_apply"] is False
    targets = {task["target"] for task in payload["tasks"]}
    assert "workflow.evidence_ingest.capture" in targets
    assert "workflow.graph_ops.materialize" in targets
    assert "workflow.network_analysis.centrality" in targets
    assert "workflow.pivot_search.query" in targets
    assert "workflow.documents_forms.packetize" in targets
    assert "workflow.runpod_artifact_import.embeddings" in targets
    for task in payload["tasks"]:
        assert task["task_type"] == "SHEET_TASK"
        assert task["status"] == "OPEN"
        assert "raw_body" not in json.dumps(task).lower()
    assert RECEIPT.exists()
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["tasks_seen"] == payload["tasks_seen"]


def test_luci_ingest_seed_workflow_tasks_hook_runs_dry():
    result = subprocess.run(
        ["./luci", "ingest", "seed-workflow-tasks", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["execution"] == "dry_run"
    assert "workflow.evidence_ingest.capture" in payload["task_targets"]
