#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_q1_model_source_records_explicit_downloaded_weight() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    source = bonsai.load_q1_model_source()
    model_path = ROOT / bonsai.Q1_MODEL_PATH
    assert model_path.is_file()
    assert source["schema"] == "lucidota.model_source.v1"
    assert source["model_id"] == "prism-ml/Bonsai-8B-gguf"
    assert source["selected_file"] == "Bonsai-8B-Q1_0.gguf"
    assert source["format"] == "GGUF Q1_0 g128 1-bit"
    assert source["expected_size_bytes"] == 1_158_654_496
    assert source["expected_size_bytes"] == model_path.stat().st_size
    assert source["expected_sha256"] == "284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54"


def test_q1_shared_runtime_config_matches_live_two_slot_vram_topology() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    runtime = bonsai.q1_shared_runtime_config()
    assert runtime["schema"] == "lucidota.local_bonsai_q1_shared.runtime.v1"
    assert runtime["model_id"] == "bonsai8b-q1-shared2"
    assert runtime["model_path"] == "03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf"
    assert runtime["slots"] == 2
    assert runtime["ctx"] == 2048
    assert runtime["ngl"] == 999
    assert runtime["kv_unified"] is True
    assert runtime["kv_offload"] is True
    assert runtime["cache_type_k"] == "q8_0"
    assert runtime["cache_type_v"] == "q8_0"
    assert runtime["base_url"] == "http://127.0.0.1:8082/v1"


def test_q1_shared_server_command_uses_single_process_two_slots_and_unified_kv() -> None:
    import lucidota_bonsai_ternary_handler as bonsai

    cmd = bonsai.build_q1_shared_server_command(binary=Path("/opt/prismml/bin/llama-server"))
    text = " ".join(map(str, cmd))
    assert cmd[0] == "/opt/prismml/bin/llama-server"
    assert "Bonsai-8B-Q1_0.gguf" in text
    assert "--parallel 2" in text
    assert "--kv-unified" in cmd
    assert "--kv-offload" in cmd
    assert "--cache-type-k q8_0" in text
    assert "--cache-type-v q8_0" in text
    assert "-ngl 999" in text
    assert "-c 2048" in text
    assert "--alias bonsai8b-q1-shared2" in text


def test_start_script_defaults_to_q1_two_slot_vram_shared_kv() -> None:
    script = (ROOT / "scripts" / "lucidota_start_bonsai_ternary_llama.sh").read_text(encoding="utf-8")
    assert 'MODEL_VARIANT="${LUCIDOTA_BONSAI_VARIANT:-q1_0}"' in script
    assert 'CTX="${LUCIDOTA_BONSAI_CTX:-2048}"' in script
    assert 'NGL="${LUCIDOTA_BONSAI_NGL:-999}"' in script
    assert '--kv-unified' in script
    assert '--cache-type-k "${LUCIDOTA_BONSAI_CACHE_TYPE_K:-q8_0}"' in script
    assert '--alias "${LUCIDOTA_BONSAI_ALIAS:-bonsai8b-q1-shared2}"' in script
