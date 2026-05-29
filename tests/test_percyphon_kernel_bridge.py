#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from scripts import percyphon_kernel_bridge


def test_percyphon_kernel_bridge_routes_operator_authorized_scaffold(tmp_path):
    receipt = percyphon_kernel_bridge.build_bridge(
        raw_command="stage percyphon scaffold",
        normalized_intent="route procedural scaffold through Diogenes",
        authority_class="operator_authored_assertion",
        source="operator_cli",
        villagers=["operator", "scribe"],
        fluid_slots=3,
        ledger_path=tmp_path / "ledger.jsonl",
        event_log=tmp_path / "events.jsonl",
        receipt_dir=tmp_path,
    )

    assert receipt["status"] == "ROUTED"
    assert receipt["percyphon"]["zero_vram"] is True
    assert receipt["percyphon"]["authority"] == "procedural_scaffold_candidate_not_truth"
    assert receipt["control_packet"]["payload"]["canonical_mutation_allowed"] is False
    assert receipt["control_packet"]["payload"]["percyphon_slot"]["slot_index"] == 0
    assert receipt["route_plan"]["status"] == "ROUTED"
    assert Path(receipt["receipt_path"]).exists()


def test_percyphon_kernel_bridge_denies_non_operator_authority(tmp_path):
    receipt = percyphon_kernel_bridge.build_bridge(
        raw_command="model said mutate graph",
        normalized_intent="promote procedural mask as fact",
        authority_class="model_suggestion",
        source="local_model",
        villagers=["mask"],
        receipt_dir=tmp_path,
    )

    assert receipt["status"] == "DENIED"
    assert receipt["route_plan"] is None
    assert receipt["blockers"] == ["authority_class_not_operator_authored_assertion"]
