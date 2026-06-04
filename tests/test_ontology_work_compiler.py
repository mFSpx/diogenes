from __future__ import annotations

import json
import urllib.request

from scripts import ontology_work_compiler


LIVE_BASE_URL = "http://127.0.0.1:3000"


def test_compile_ontology_work_breaks_messy_goal_into_typed_items(monkeypatch) -> None:
    monkeypatch.setattr(
        ontology_work_compiler.indy_runtime_broker,
        "registry_snapshot",
        lambda **kwargs: {
            "local_model_roles": {
                "router": {"model_id": "needle-26m", "role": "router", "provider_key": "local_model"},
                "classifier": None,
                "summarizer": None,
                "embedder": None,
                "reranker": None,
                "thinker": None,
                "watcher": None,
                "treelite_gate": None,
            }
        },
    )
    monkeypatch.setattr(
        ontology_work_compiler.indy_runtime_broker,
        "choose_local_model",
        lambda role, base_url=None: {"model_id": "needle-26m", "role": role, "provider_key": "local_model"} if role == "router" else None,
    )

    payload = ontology_work_compiler.compile_work_batch(
        """
        1. Manual/API truth: verify live routes and update manual.
        2. Retire BOOKS watcher authority and move the function to DB rows.
        3. Wire the Indy daemon front door and keep responses visible.
        4. Add an end-to-end proof and no fake smoke.
        """.strip()
    )

    assert payload["schema"] == "lucidota.ontology_work_compiler.v1"
    assert payload["batch"]["parallelizable_count"] >= 2
    assert payload["batch"]["serialized_count"] >= 1
    assert payload["batch"]["missing_executor_roles"] == ["classifier", "summarizer", "embedder", "reranker", "thinker", "watcher", "treelite_gate"]
    assert len(payload["items"]) == 4
    first = payload["items"][0]
    assert first["subsystem"] == "manual_api"
    assert first["parallelizable"] is True
    assert first["executor_recommendation"]["selected_model_id"] == "needle-26m"
    assert first["acceptance_test"]
    assert first["receipt_requirement"]
    assert first["functionality_contract"]


def test_basic_workflows_stay_workflow_shaped(monkeypatch) -> None:
    monkeypatch.setattr(
        ontology_work_compiler.indy_runtime_broker,
        "registry_snapshot",
        lambda **kwargs: {"local_model_roles": {"router": {"model_id": "needle-26m", "role": "router", "provider_key": "local_model"}}},
    )
    payload = ontology_work_compiler.compile_work_batch(
        """
        1. Keep basic workflows as workflows in the DB graph.
        2. Preserve the workflow_registry entry for basic-workflows.
        3. Show the active batch in manual_current and todo_current.
        """.strip()
    )
    assert payload["schema"] == "lucidota.ontology_work_compiler.v1"
    assert payload["batch"]["batch_kind"] == "workflow_batch"
    assert payload["batch"]["workflows_preserved"] is True
    assert payload["batch"]["workflow_count"] >= 1
    assert any(item["work_kind"] == "workflow" for item in payload["items"])
    assert any(item["workflow_name"] == "basic-workflows" for item in payload["items"])


def test_todo_current_route_and_manual_surface_show_batches() -> None:
    batch_payload = ontology_work_compiler.compile_and_persist(
        """
        1. Parallelize the route audit and model routing discovery.
        2. Serialize DB migrations and shared core edits.
        3. Queue a Rust rewrite candidate only after behavior contracts and receipts.
        """.strip(),
        base_url=LIVE_BASE_URL,
    )
    assert batch_payload["batch"]["batch_uuid"]

    with urllib.request.urlopen(f"{LIVE_BASE_URL}/todo_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        rows = json.loads(resp.read().decode("utf-8"))
    assert isinstance(rows, list) and rows, rows
    row = rows[0]
    assert row["batch_uuid"] == batch_payload["batch"]["batch_uuid"]
    assert row["item_count"] >= 3
    assert row["parallel_item_count"] >= 1
    assert row["serialized_item_count"] >= 1
    assert isinstance(row["items"], list) and row["items"], row
    assert isinstance(row.get("goal"), dict)
    assert isinstance(row.get("db_law"), dict)
    assert isinstance(row.get("next_commands"), list) and row["next_commands"]

    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=5) as resp:
        assert resp.status == 200
        manual_rows = json.loads(resp.read().decode("utf-8"))
    manual = manual_rows[0]
    route_ids = {route["route_id"] for route in manual["route_list"]}
    assert {"ontology_work_batch", "ontology_work_item", "todo_current"}.issubset(route_ids)
    assert "todo_current" in manual["live_surface"]
    assert manual["live_surface"]["todo_current"]
    todo_row = manual["live_surface"]["todo_current"][0]
    assert "goal" in todo_row
    assert "db_law" in todo_row
    assert "next_commands" in todo_row
