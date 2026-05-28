#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_swarm_usage_ledger_aggregates_local_groq_cohere_and_main(tmp_path):
    import swarm_usage_ledger

    local = write_json(
        tmp_path / "local.json",
        {
            "schema": "lucidota.model_invocation.local_chat.v1",
            "provider": "local",
            "lane": "mamba_gpu",
            "status": "PASS",
            "token_accounting": {"total_tokens": 8, "source": "provider_usage"},
        },
    )
    groq = write_json(
        tmp_path / "groq.json",
        {
            "schema": "lucidota.model_invocation.groq_chat.v1",
            "provider": "groq",
            "status": "PASS",
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        },
    )
    blocked = write_json(
        tmp_path / "blocked.json",
        {"schema": "lucidota.model_invocation.groq_chat.v1", "provider": "groq", "status": "BLOCKED", "blockers": ["groq_http_error:401"]},
    )
    cohere = write_json(
        tmp_path / "cohere.json",
        {
            "schema": "lucidota.model_invocation.cohere_chat.v1",
            "provider": "cohere",
            "status": "PASS",
            "raw_response": {"usage": {"tokens": {"input_tokens": 7, "output_tokens": 5}}},
        },
    )

    ledger = swarm_usage_ledger.build_ledger([local, groq, blocked, cohere], main_tokens=25)

    assert ledger["totals"]["local"]["tokens"] == 8
    assert ledger["totals"]["groq"]["tokens"] == 10
    assert ledger["totals"]["cohere"]["tokens"] == 12
    assert ledger["totals"]["main_agent"]["tokens"] == 25
    assert ledger["totals"]["all_accounted_tokens"] == 55
    assert ledger["target_policy"]["main_agent"]["target_share"] == 0.10
    assert ledger["blockers"][0]["blockers"] == ["groq_http_error:401"]
    assert ledger["totals"]["groq"]["share"] == 10 / 55
    assert ledger["totals"]["cohere"]["share"] == 12 / 55


def test_swarm_usage_ledger_policy_matches_current_goal_split():
    import swarm_usage_ledger

    ledger = swarm_usage_ledger.build_ledger([])

    assert {lane: policy["target_share"] for lane, policy in ledger["target_policy"].items()} == {
        "main_agent": 0.10,
        "groq": 0.35,
        "cohere": 0.15,
        "local": 0.40,
    }
    assert sum(policy["target_share"] for policy in ledger["target_policy"].values()) == 1.0


def test_swarm_usage_ledger_discovers_model_invocation_receipts_by_default(tmp_path):
    import swarm_usage_ledger

    receipt_dir = tmp_path / "05_OUTPUTS" / "model_invocations"
    receipt_dir.mkdir(parents=True)
    groq = write_json(
        receipt_dir / "groq_chat_execute_20260527T000000Z.json",
        {
            "schema": "lucidota.model_invocation.groq_chat.v1",
            "provider": "groq",
            "status": "PASS",
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
    )
    write_json(receipt_dir / "project2501_model_workshare_observation_latest.json", {"schema": "lucidota.not_a_model_invocation.v1"})

    receipts = swarm_usage_ledger.discover_receipts(root=tmp_path)
    ledger = swarm_usage_ledger.build_ledger(receipts)

    assert receipts == [groq]
    assert ledger["discovery"]["mode"] == "auto_model_invocations"
    assert ledger["totals"]["groq"]["tokens"] == 5
