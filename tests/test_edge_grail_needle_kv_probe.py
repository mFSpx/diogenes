import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/edge_grail_needle_kv_probe.py")


def test_needle_kv_probe_reports_encoder_decoder_truth_and_boundary(tmp_path):
    receipt = tmp_path / "needle_kv.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--receipt", str(receipt), "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.edge_grail.needle_kv_probe.v1"
    assert payload["status"] == "PARTIAL_CURRENT_RUNNER_BATCHES_PREFIX_REFACTOR_REQUIRED"
    assert payload["architecture"]["kind"] == "jax_encoder_decoder_transformer"
    assert payload["architecture"]["has_separate_encode_method"] is True
    assert payload["architecture"]["has_separate_decode_method"] is True
    assert payload["current_worker"]["one_process_six_lane_batch"] is True
    assert payload["truth_flags"]["shared_weights_one_process_proven_by_source"] is True
    assert payload["truth_flags"]["batched_shared_prefix_available"] is True
    assert payload["truth_flags"]["exact_tensor_kv_pointer_sharing_currently_proven"] is False
    assert payload["truth_flags"]["prefix_reuse_possible_for_identical_full_encoder_input"] is True
    assert payload["truth_flags"]["prefix_reuse_for_different_lane_tools_requires_refactor"] is True
    assert payload["next_runner_patch"]["required"] is True
    assert receipt.exists()


def test_needle_worker_health_exposes_kv_probe_receipt_path():
    worker = Path("scripts/lucidota_needle_worker.py").read_text(encoding="utf-8")
    assert "needle_kv_probe_latest.json" in worker
    assert "exact_tensor_kv_pointer_sharing_currently_proven" in worker
    assert "prefix_reuse_for_different_lane_tools_requires_refactor" in worker
