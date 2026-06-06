from __future__ import annotations

import json
import urllib.error
import urllib.request

import psycopg


LIVE_BASE_URL = "http://127.0.0.1:3000"
LIVE_DSN = "postgresql:///lucidota_state"


def get_json(path: str) -> object:
    with urllib.request.urlopen(f"{LIVE_BASE_URL}/{path}", timeout=5) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def post_json(path: str, payload: dict[str, object]) -> object:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LIVE_BASE_URL}/{path}",
        data=data,
        headers={
            "content-type": "application/json",
            "accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def test_prompt_routes_are_readable_and_manual_mentions_prompt_ledger() -> None:
    for route in [
        "prompts_filed?limit=1",
        "prompt_work_order_links?limit=1",
        "prompt_recent?limit=1",
        "prompt_unlinked?limit=1",
        "prompt_catalog_status?limit=1",
    ]:
        payload = get_json(route)
        assert isinstance(payload, list), (route, payload)

    manual = get_json("manual_current?limit=1")
    row = manual[0]
    route_ids = {route["route_id"] for route in row["route_list"]}
    assert {
        "prompts_filed",
        "prompt_work_order_links",
        "prompt_recent",
        "prompt_unlinked",
        "prompt_catalog_status",
        "file_prompt",
        "link_prompt_work_order",
        "decompose_prompt_to_work_orders",
    }.issubset(route_ids)

    prompt_catalog_rows = get_json("prompt_catalog_status?limit=1")
    assert isinstance(prompt_catalog_rows, list) and prompt_catalog_rows, prompt_catalog_rows
    row = prompt_catalog_rows[0]
    packet = row["packet"]
    assert isinstance(packet.get("next_commands"), list) and packet["next_commands"]
    assert isinstance(packet.get("next_command_refs"), list) and packet["next_command_refs"]
    assert all(not str(cmd).startswith("./luci") for cmd in packet["next_commands"])
    assert "prompt_catalog_status" in packet["next_commands"]
    assert "prompt_catalog_status" in packet["next_command_refs"]
    assert "api_prompt_catalog_status" in packet["next_command_refs"]


def test_file_prompt_is_idempotent_and_raw_text_is_preserved() -> None:
    payload = {
        "source": "operator",
        "source_model": "codex",
        "receiving_model": "indy_reads",
        "target_model": "indy_reads",
        "raw_prompt_text": "Capture the current steering prompts and file them into the DB.",
        "normalized_prompt_text": "Capture the current steering prompts and file them into the DB.",
        "conversation_session_id": "session-test-prompt-ledger",
        "linked_goal_id": "root-orchestrator-current",
        "ontology_tags": ["GO", "IO", "prompt-ledger"],
        "subsystem_tags": ["manual", "api"],
        "notes": "surface test",
        "blockers": "",
        "idempotency_key": "prompt-ledger-test-1",
    }

    first = post_json("rpc/file_prompt", payload)
    second = post_json("rpc/file_prompt", payload)

    assert isinstance(first, dict) and isinstance(second, dict)
    assert first["prompt_id"] == second["prompt_id"]
    assert first["prompt_hash"] == second["prompt_hash"]
    assert first["raw_prompt_text"] == payload["raw_prompt_text"]
    assert first["normalized_prompt_text"] == payload["normalized_prompt_text"]

    prompt_rows = get_json(f"prompts_filed?prompt_id=eq.{first['prompt_id']}")
    assert isinstance(prompt_rows, list) and len(prompt_rows) == 1
    assert prompt_rows[0]["raw_prompt_text"] == payload["raw_prompt_text"]
    assert prompt_rows[0]["idempotency_key"] == payload["idempotency_key"]

    unlinked_rows = get_json(f"prompt_unlinked?prompt_id=eq.{first['prompt_id']}")
    assert isinstance(unlinked_rows, list) and len(unlinked_rows) == 1
    assert unlinked_rows[0]["prompt_id"] == first["prompt_id"]


def test_prompt_link_and_decompose_are_visible_through_routes() -> None:
    with psycopg.connect(LIVE_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select work_order_uuid::text from lucidota_control.work_order order by created_at desc limit 1"
            )
            work_order_uuid = cur.fetchone()[0]

    base_payload = {
        "source": "codex",
        "source_model": "codex",
        "receiving_model": "indy_reads",
        "target_model": "indy_reads",
        "raw_prompt_text": "Split the queue into parallel work batches and expose the plan in the manual.",
        "normalized_prompt_text": "Split the queue into parallel work batches and expose the plan in the manual.",
        "conversation_session_id": "session-test-prompt-ledger-2",
        "linked_goal_id": "root-orchestrator-current",
        "ontology_tags": ["GO", "CO", "IO"],
        "subsystem_tags": ["orchestrator", "manual"],
        "idempotency_key": "prompt-ledger-test-2",
    }

    filed = post_json("rpc/file_prompt", base_payload)
    linked = post_json(
        "rpc/link_prompt_work_order",
        {
            "p_prompt_id": filed["prompt_id"],
            "p_work_order_uuid": work_order_uuid,
            "p_linked_goal_id": "root-orchestrator-current",
        },
    )
    assert linked["prompt_id"] == filed["prompt_id"]
    assert work_order_uuid in linked["linked_work_order_uuid"]

    link_rows = get_json(f"prompt_work_order_links?prompt_id=eq.{filed['prompt_id']}")
    assert isinstance(link_rows, list) and link_rows
    assert any(row["work_order_uuid"] == work_order_uuid for row in link_rows)

    decomposed = post_json("rpc/decompose_prompt_to_work_orders", {"prompt_id": filed["prompt_id"]})
    assert decomposed["prompt_id"] == filed["prompt_id"]
    assert isinstance(decomposed.get("subsystem_tags"), list) and decomposed["subsystem_tags"]
    assert isinstance(decomposed.get("ontology_tags"), list) and decomposed["ontology_tags"]
    assert isinstance(decomposed.get("linked_work_order_uuid"), list) and decomposed["linked_work_order_uuid"]

    catalog_rows = get_json("prompt_catalog_status?limit=1")
    assert isinstance(catalog_rows, list) and catalog_rows
    assert catalog_rows[0]["prompt_count"] >= 2
    assert catalog_rows[0]["linked_count"] >= 1


def test_prompt_records_do_not_keep_old_curl_acceptance_tests() -> None:
    rows = get_json("prompts_filed?limit=25")
    assert isinstance(rows, list) and rows
    assert all(
        "curl the live route" not in json.dumps(row).lower()
        for row in rows
    )
