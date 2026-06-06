from __future__ import annotations

import json
from pathlib import Path


def test_indy_reads_boot_packet_includes_bonsai_dual_slot_load_plan() -> None:
    packet = json.loads(Path("04_RUNTIME/indy_reads_boot_packet.json").read_text(encoding="utf-8"))

    assert packet["schema"] == "lucidota.indy_reads.boot_packet.v1"
    model_fabric = packet["model_fabric"]
    assert model_fabric["primary_cortex"]["model_lane"] == "bonsai_q1_0"
    assert model_fabric["primary_cortex"]["shared_weight"] is True
    assert model_fabric["primary_cortex"]["logical_slots"] == ["slot_0", "slot_1"]
    assert model_fabric["primary_cortex"]["slot_0_role"] == "synthesis"
    assert model_fabric["primary_cortex"]["slot_1_role"] == "skeptic_verifier"
    assert model_fabric["primary_cortex"]["default_context_tokens"] == 10_000
    assert model_fabric["primary_cortex"]["max_context_tokens_after_proof"] == 16_000
    assert model_fabric["primary_cortex"]["prefix_cache_required"] is True
    assert model_fabric["primary_cortex"]["quantized_kv_preferred"] is True
    assert model_fabric["reflex_bank"]["shared_weight"] is True
    assert model_fabric["reflex_bank"]["logical_lanes"] == 6
    assert model_fabric["state_watcher"]["role"] == "state_flow_watcher"
    assert model_fabric["recursive_bank"]["logical_workers"] == 20
    assert model_fabric["rolling_language"]["model_lane"] == "rwkv_world_400m"
