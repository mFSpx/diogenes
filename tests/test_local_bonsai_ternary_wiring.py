#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_bonsai_runtime_points_only_at_local_ternary_bonsai() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    runtime = bonsai.default_runtime_config()
    assert runtime["model_id"] == "Ternary-Bonsai-4B-Q2_0.gguf"
    assert runtime["model_path"] == "03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf"
    assert runtime["handler"] == "llama.cpp-prismml-q2_0"
    assert runtime["base_url"] == "http://127.0.0.1:8082/v1"
    assert all(forbidden not in json.dumps(runtime).lower() for forbidden in ["mamba", "deepseek"])


def test_bonsai_model_source_matches_local_file_without_download() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    source = bonsai.load_model_source()
    model_path = ROOT / bonsai.MODEL_PATH
    assert model_path.is_file()
    assert source["format"] == "GGUF Q2_0 g128 ternary {-1,0,+1}"
    assert source["selected_file"] == model_path.name
    assert source["expected_size_bytes"] == model_path.stat().st_size
    assert source["base_model"] == "Qwen3-4B"


def test_bonsai_server_command_uses_custom_prismml_llama_binary() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    cmd = bonsai.build_server_command(binary=Path("/opt/prismml/bin/llama-server"), port=8899, ctx=512, ngl=0)
    text = " ".join(map(str, cmd))
    assert cmd[0] == "/opt/prismml/bin/llama-server"
    assert "Ternary-Bonsai-4B-Q2_0.gguf" in text
    assert "--port 8899" in text
    assert "-c 512" in text
    assert "-ngl 0" in text
    assert "--no-warmup" in text


def test_llxprt_provider_config_targets_bonsai_openai_compatible_server() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    provider = bonsai.llxprt_provider_config()
    profile = bonsai.llxprt_profile_config()
    assert provider["name"] == "lucidota-local"
    assert provider["base-url"] == "http://127.0.0.1:8082/v1"
    assert provider["defaultModel"] == "Ternary-Bonsai-4B-Q2_0.gguf"
    assert provider["staticModels"] == [{"id": "Ternary-Bonsai-4B-Q2_0.gguf", "name": "Ternary Bonsai 4B Q2_0 local"}]
    assert profile["provider"] == "lucidota-local"
    assert profile["model"] == "Ternary-Bonsai-4B-Q2_0.gguf"


def test_bonsai_prefers_cuda_prismml_build_for_speed() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    candidates = [str(p) for p in bonsai.llama_server_candidates()]
    assert candidates[0].endswith("01_REPOS/prismml_llama.cpp/build-cuda/bin/llama-server")
    assert candidates[1].endswith("01_REPOS/prismml_llama.cpp/build/bin/llama-server")
