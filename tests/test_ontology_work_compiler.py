from __future__ import annotations

import json
import sys
import urllib.request

import pytest

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


def test_sql_bind_guard_matches_persist_batch_queries(monkeypatch) -> None:
    executed: list[tuple[str, tuple[object, ...]]] = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            executed.append((sql, tuple(params)))

        def fetchone(self):
            return ("11111111-1111-1111-1111-111111111111",)

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            return None

    fake_psycopg = type("FakePsycopg", (), {"connect": lambda self, db_url, connect_timeout=5: FakeConn()})()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    payload = {
        "schema": "lucidota.ontology_work_compiler.v1",
        "generated_at": "2026-06-04T00:00:00Z",
        "batch": {
            "batch_key": "ontobatch:test",
            "source_ref": "operator_turn",
            "source_kind": "operator_text",
            "source_hash": "hash",
            "source_excerpt": "excerpt",
            "objective_summary": "objective",
            "subsystem": "mixed",
            "ontology_tags": ["TEST"],
            "dependency_edges": [],
            "risk": "medium",
            "parallel_policy": "mixed",
            "planner_groups": [],
            "selected_lanes": [],
            "missing_executor_roles": [],
            "executor_recommendation": {"status": "ready"},
            "acceptance_test": "accept",
            "receipt_requirement": "receipt",
            "functionality_contract": "contract",
            "workflow_count": 1,
            "workflows_preserved": True,
            "batch_kind": "workflow_batch",
            "status": "draft",
            "detail": {"notes": []},
        },
        "items": [
            {
                "item_rank": 1,
                "planner_group": "parallel_scan",
                "work_kind": "audit",
                "workflow_name": "manual",
                "subsystem": "manual_api",
                "ontology_tags": ["TEST"],
                "dependency_edges": [],
                "risk": "low",
                "parallelizable": True,
                "serialized": False,
                "route_hint": "/manual_current",
                "executor_recommendation": {"status": "ready"},
                "acceptance_test": "accept",
                "receipt_requirement": "receipt",
                "functionality_contract": "contract",
                "status": "draft",
                "detail": {"notes": []},
            }
        ],
        "selected_lanes": [],
        "missing_executor_roles": [],
    }

    result = ontology_work_compiler.persist_batch(payload, db_url="postgresql://fake")
    assert result["batch"]["batch_uuid"] == "11111111-1111-1111-1111-111111111111"
    assert len(executed) == 3
    for sql, params in executed:
        assert ontology_work_compiler.sql_placeholder_count(sql) == len(params)


def test_sql_bind_guard_raises_on_mismatch() -> None:
    class DummyCursor:
        def execute(self, sql, params):  # pragma: no cover - should not be reached
            raise AssertionError("execute should not be called on mismatch")

    with pytest.raises(ValueError, match="placeholder_count=2 bind_count=1"):
        ontology_work_compiler.execute_with_bind_guard(DummyCursor(), "SELECT %s, %s", (1,))


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
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "manual_current" in row["next_command_refs"]
    assert "root_orchestrator_current" in row["next_command_refs"]
    assert "command_registry" in row["next_command_refs"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert row["orchestration"]["strict_priority_stack"][0] == "live_truth_surfaces"

    with urllib.request.urlopen(f"{LIVE_BASE_URL}/manual_current?limit=1", timeout=15) as resp:
        assert resp.status == 200
        manual_rows = json.loads(resp.read().decode("utf-8"))
    manual = manual_rows[0]
    route_ids = {route["route_id"] for route in manual["route_list"]}
    assert {"ontology_work_batch", "ontology_work_item", "todo_current"}.issubset(route_ids)
    assert "todo_current" in manual["live_surface"]
    assert manual["live_surface"]["todo_current"]


def test_todo_current_packets_do_not_keep_old_curl_acceptance_tests() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/todo_current?limit=25", timeout=5) as resp:
        assert resp.status == 200
        rows = json.loads(resp.read().decode("utf-8"))
    assert isinstance(rows, list) and rows
    assert all(
        "curl the live route" not in json.dumps(row).lower()
        for row in rows
    )


def test_ontology_work_packets_do_not_keep_old_curl_acceptance_tests() -> None:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/ontology_work_batch?limit=25", timeout=5) as resp:
        assert resp.status == 200
        batch_rows = json.loads(resp.read().decode("utf-8"))
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/ontology_work_item?limit=25", timeout=5) as resp:
        assert resp.status == 200
        item_rows = json.loads(resp.read().decode("utf-8"))

    assert isinstance(batch_rows, list) and batch_rows
    assert isinstance(item_rows, list) and item_rows
    assert all("curl the live route" not in json.dumps(row).lower() for row in batch_rows)
    assert all("curl the live route" not in json.dumps(row).lower() for row in item_rows)
