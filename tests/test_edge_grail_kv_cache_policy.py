import json
import subprocess
import sys
from pathlib import Path

POLICY = Path("04_RUNTIME/edge_grail_kv_cache_policy.json")
SCRIPT = Path("scripts/edge_grail_kv_cache_policy.py")
RECEIPT = Path("05_OUTPUTS/runtime/edge_grail_kv_cache_policy_latest.json")


def test_kv_policy_manifest_uses_rolling_500_token_windows_not_flush_every_500():
    data = json.loads(POLICY.read_text(encoding="utf-8"))
    assert data["schema"] == "lucidota.edge_grail.kv_cache_policy.v1"
    assert data["policy"] == "rolling_window_shared_prefix"
    assert data["chunk_tokens"] == 500
    assert data["flush_strategy"] == "evict_after_receipt_not_blind_flush"
    assert data["needles"]["lanes"] == 6
    assert data["needles"]["shared_prefix_tokens"] == 500
    assert data["needles"]["kv_cache_scope"] == "one_immutable_chunk_prefix_shared_by_6_lane_batch"
    assert data["needles"]["exact_tensor_kv_dedup_status"] == "RUNNER_EXTENSION_TARGET_NOT_CURRENTLY_PROVEN"
    assert data["bonsai"]["parallel_slots"] == 2
    assert data["bonsai"]["kv_unified"] is True
    assert data["bonsai"]["cache_type_k"] == "q8_0"
    assert data["bonsai"]["cache_type_v"] == "q8_0"
    assert data["treelite"]["fil_gpu_residency_proven"] is True
    assert data["treelite"]["fil_receipt"] == "05_OUTPUTS/model_runtime/treelite_fil_residency_all_tl_latest.json"
    assert data["treelite"]["tl_artifacts_tested"] == 103
    assert data["eviction"]["drop_raw_text_from_prompt"] is True
    assert data["eviction"]["never_keep_unbounded_history"] is True
    assert data["truth_law"]["mamba_literal_q1_0_absent"] is True
    assert data["truth_law"]["mamba_iq1_s_one_bit_class_downloaded"] is True
    assert data["truth_law"]["treelite_gpu_residency_proven_by_runpod_fil_receipt"] is True
    assert data["receipt_required"] is True


def test_kv_policy_script_writes_receipt_and_ledger():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["policy"] == "rolling_window_shared_prefix"
    assert payload["chunk_tokens"] == 500
    assert payload["ledger_mib"]["target_total_allocated_mib"] <= 3714
    assert payload["ledger_mib"]["target_remaining_mib"] > 0
    assert payload["truth_flags"]["exact_needle_tensor_kv_dedup_proven"] is False
    assert payload["truth_flags"]["shared_needle_server_batched_prefix_available"] is True
    assert payload["truth_flags"]["mamba_iq1_s_one_bit_class_downloaded"] is True
    assert payload["truth_flags"]["mamba_literal_q1_0_downloaded"] is False
    assert payload["truth_flags"]["treelite_gpu_residency_proven"] is True
    assert RECEIPT.exists()
