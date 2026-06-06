import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_bonsai_8b_q2_and_needle_have_authoritative_model_source_manifests():
    bonsai = read_json("03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/model_source.json")
    assert bonsai["schema"] == "lucidota.model_source.v1"
    assert bonsai["model_id"] == "prism-ml/Ternary-Bonsai-8B-gguf"
    assert bonsai["selected_file"] == "Ternary-Bonsai-8B-Q2_0.gguf"
    assert bonsai["format"] == "GGUF Q2_0 g128 ternary {-1,0,+1}"
    assert bonsai["expected_size_bytes"] == (ROOT / "03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf").stat().st_size
    assert bonsai["expected_sha256"] == "3c8d70470a5d97e5a2b9410ddd899cb740116591462626c60cb2fead6448f60b"

    needle = read_json("03_VAULT/models/needle/model_source.json")
    assert needle["schema"] == "lucidota.model_source.v1"
    assert needle["model_id"] == "cactus-compute/needle"
    assert needle["selected_file"] == "needle.pkl"
    assert needle["parameter_count"] == 26_000_000
    assert needle["expected_size_bytes"] == (ROOT / "03_VAULT/models/needle/needle.pkl").stat().st_size
    assert needle["expected_sha256"] == "40a32e91d1d4197bf15ba559b74f6727c342dc8746918742fc7d8e2c1f18df40"


def test_edge_runtime_status_points_to_all_downloaded_weight_custody_manifests():
    status = read_json("05_OUTPUTS/runtime/edge_grail_runtime_status_latest.json")
    manifests = status["source_manifests"]
    assert manifests["bonsai_8b_q1_0"] == "03_VAULT/models/prism-ml/Bonsai-8B-gguf/model_source.json"
    assert manifests["bonsai_8b_q2_0"] == "03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/model_source.json"
    assert manifests["needle_26m"] == "03_VAULT/models/needle/model_source.json"
    weights = status["verified_weights"]
    assert weights["bonsai_8b_q2_0_cpu_ternary"]["present"] is True
    assert weights["needle_26m"]["present"] is True
