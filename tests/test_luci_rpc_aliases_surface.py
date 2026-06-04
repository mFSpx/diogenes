from __future__ import annotations

import json
import subprocess
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]
LIVE_DSN = "postgresql:///lucidota_state"


def test_luci_rpc_file_prompt_decompose_and_link_aliases_are_live() -> None:
    base_payload = {
        "source": "codex",
        "source_model": "codex",
        "receiving_model": "indy_reads",
        "target_model": "indy_reads",
        "raw_prompt_text": "Promote the alias RPC surfaces into the operator cockpit.",
        "normalized_prompt_text": "Promote the alias RPC surfaces into the operator cockpit.",
        "conversation_session_id": "session-rpc-alias-test",
        "linked_goal_id": "root-orchestrator-current",
        "ontology_tags": ["GO", "CO", "IO"],
        "subsystem_tags": ["manual", "api"],
        "notes": "rpc alias surface test",
        "blockers": "",
        "idempotency_key": "rpc-alias-test-1",
    }

    file_proc = subprocess.run(
        [str(ROOT / "luci"), "rpc", "file-prompt", "--payload-json", json.dumps(base_payload), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    filed = json.loads(file_proc.stdout)
    assert filed["source_url"].endswith("/rpc/file_prompt")
    filed_payload = filed["payload"]
    assert filed_payload["prompt_id"]
    assert filed_payload["raw_prompt_text"] == base_payload["raw_prompt_text"]

    with psycopg.connect(LIVE_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select work_order_uuid::text from lucidota_control.work_order order by created_at desc limit 1"
            )
            work_order_uuid = cur.fetchone()[0]

    link_proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "rpc",
            "link-prompt",
            "--payload-json",
            json.dumps(
                {
                    "p_prompt_id": filed_payload["prompt_id"],
                    "p_work_order_uuid": work_order_uuid,
                    "p_linked_goal_id": "root-orchestrator-current",
                }
            ),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    linked = json.loads(link_proc.stdout)
    linked = linked["payload"]
    assert linked["prompt_id"] == filed_payload["prompt_id"]
    assert work_order_uuid in linked["linked_work_order_uuid"]

    decompose_proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "rpc",
            "decompose-prompt",
            "--payload-json",
            json.dumps({"prompt_id": filed_payload["prompt_id"]}),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    decomposed = json.loads(decompose_proc.stdout)["payload"]
    assert decomposed["prompt_id"] == filed_payload["prompt_id"]
    assert isinstance(decomposed.get("subsystem_tags"), list) and decomposed["subsystem_tags"]
    assert isinstance(decomposed.get("ontology_tags"), list) and decomposed["ontology_tags"]


def test_luci_rpc_cloud_packet_alias_is_live() -> None:
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "rpc",
            "cloud-packet",
            "--work-order-id",
            "00000000-0000-0000-0000-000000000000",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert "source_url" in payload
    assert payload["source_url"].endswith("/rpc/cloud_packet")
