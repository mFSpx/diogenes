from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from graph_materialization_helper import evaluate_command_materialization_policy


def command(**overrides):
    base = {
        "command_uuid": "00000000-0000-0000-0000-000000000001",
        "status": "queued",
        "allowed_effect": "canonical graph materialization through graph_promoter transaction",
        "authority_class": "operator_authored_assertion",
        "canonical_mutation_allowed": False,
        "conversation_required": True,
        "evidence_refs": ["tests/test_graph_materialization_command_policy.py"],
        "target_refs": ["graph_item:policy_smoke"],
        "command_envelope": {
            "protocol": "lucidota.surface_instruction_envelope.v1",
            "staging_only": False,
            "canonical_mutation_allowed": False,
            "allowed_effect": "canonical graph materialization through graph_promoter transaction",
            "graph_materialization_policy": "graph_promoter_transaction",
        },
    }
    base.update(overrides)
    return base


def test_materialization_policy_allows_explicit_graph_promoter_transaction():
    decision = evaluate_command_materialization_policy(command())

    assert decision["allowed"] is True
    assert decision["blockers"] == []


def test_materialization_policy_rejects_no_graph_materialization_command():
    decision = evaluate_command_materialization_policy(
        command(
            status="executed",
            allowed_effect="queue DBOS work order only; no graph materialization",
            target_refs=[],
            command_envelope={
                "staging_only": True,
                "canonical_mutation_allowed": False,
                "allowed_effect": "queue DBOS work order only; no graph materialization",
            },
        )
    )

    assert decision["allowed"] is False
    assert "allowed_effect_explicitly_forbids_graph_materialization" in decision["blockers"]
    assert "command_envelope_graph_materialization_policy_required" in decision["blockers"]


def test_materialization_policy_rejects_staged_or_evidence_free_commands():
    decision = evaluate_command_materialization_policy(
        command(status="staged", evidence_refs=[], command_envelope={})
    )

    assert decision["allowed"] is False
    assert "command_status_must_be_queued_accepted_or_executed" in decision["blockers"]
    assert "command_evidence_refs_required" in decision["blockers"]
    assert "command_envelope_graph_materialization_policy_required" in decision["blockers"]
