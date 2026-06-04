import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/edge_grail_local_admission.py")
POLICY = Path("04_RUNTIME/edge_grail_kv_cache_policy.json")


def test_evaluate_admission_distinguishes_now_from_after_spindown():
    sys.path.insert(0, str(Path("scripts").resolve()))
    import edge_grail_local_admission as admission

    policy = {
        "ledger_mib": {
            "target_total_allocated_mib": 3210,
            "target_remaining_mib": 504,
            "gtx1650_budget_mib": 3714,
        }
    }
    current = admission.evaluate_admission(
        policy,
        gpu={"name": "NVIDIA GeForce GTX 1650", "total_mib": 4096, "used_mib": 1264, "free_mib": 2451, "temperature_c": 56},
        reserve_mib=450,
    )
    assert current["status"] == "ADMISSIBLE_AFTER_SPINDOWN"
    assert current["truth_flags"]["fits_card_with_reserve"] is True
    assert current["truth_flags"]["admissible_now"] is False
    assert current["truth_flags"]["requires_spindown"] is True
    assert current["needed_free_mib"] == 3660

    now = admission.evaluate_admission(
        policy,
        gpu={"name": "NVIDIA GeForce GTX 1650", "total_mib": 4096, "used_mib": 200, "free_mib": 3896, "temperature_c": 45},
        reserve_mib=450,
    )
    assert now["status"] == "ADMISSIBLE_NOW"
    assert now["truth_flags"]["admissible_now"] is True
    assert now["truth_flags"]["requires_spindown"] is False


def test_local_admission_cli_writes_receipt_without_starting_models(tmp_path):
    receipt = tmp_path / "admission.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--policy",
            str(POLICY),
            "--receipt",
            str(receipt),
            "--mock-gpu",
            "NVIDIA GeForce GTX 1650,4096,1264,2451,56",
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.edge_grail.local_admission.v1"
    assert payload["status"] == "ADMISSIBLE_AFTER_SPINDOWN"
    assert payload["models_started"] is False
    assert payload["models_killed"] is False
    assert payload["policy"]["path"] == str(POLICY)
    assert payload["ledger_mib"]["target_total_allocated_mib"] == 3210
    assert receipt.exists()
