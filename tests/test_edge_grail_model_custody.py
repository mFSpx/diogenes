import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_mamba_one_bit_class_weight_is_downloaded_and_not_overclaimed_as_q1_0():
    source = read_json("03_VAULT/models/mradermacher/Falcon3-Mamba-7B-Instruct-i1-GGUF/model_source.json")
    model = ROOT / "03_VAULT/models/mradermacher/Falcon3-Mamba-7B-Instruct-i1-GGUF/Falcon3-Mamba-7B-Instruct.i1-IQ1_S.gguf"
    assert model.is_file()
    assert source["schema"] == "lucidota.model_source.v1"
    assert source["selected_file"] == model.name
    assert source["expected_size_bytes"] == model.stat().st_size
    assert source["sha256"] == "d198991367c2a95a882b7b39061f0822ce34e0e0bca45fef96a8a1dbd948a973"
    assert "IQ1_S" in source["format"]
    assert "NOT an explicit Q1_0" in source["format"]


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
    assert manifests["mamba_7b_iq1_s_one_bit_class"] == "03_VAULT/models/mradermacher/Falcon3-Mamba-7B-Instruct-i1-GGUF/model_source.json"
    assert manifests["mamba_7b_q2_q3_fallback"] == "03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/model_source.json"
    assert manifests["needle_26m"] == "03_VAULT/models/needle/model_source.json"
    weights = status["verified_weights"]
    assert weights["mamba_7b_iq1_s_one_bit_class"]["present"] is True
    assert weights["mamba_7b_q1_or_1bit"]["present"] is False
    assert weights["bonsai_8b_q2_0_cpu_ternary"]["present"] is True
    assert weights["needle_26m"]["present"] is True
