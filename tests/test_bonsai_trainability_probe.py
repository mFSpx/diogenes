from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bonsai_trainability_probe.py"
MODEL_SOURCE = ROOT / "03_VAULT" / "models" / "prism-ml" / "Ternary-Bonsai-8B-gguf" / "model_source.json"


def load_probe_module():
    assert SCRIPT.exists(), f"missing probe script: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("bonsai_trainability_probe", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_parses_loader_failure_and_keeps_exact_target_blocked(tmp_path: Path) -> None:
    probe = load_probe_module()
    loader_log = tmp_path / "bonsai_standard_cuda_probe.log"
    loader_log.write_text(
        "\n".join(
            [
                "main: loading model",
                "srv    load_model: loading model '03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf'",
                "gguf_init_from_file_ptr: tensor 'token_embd.weight' has invalid ggml type 42. should be in [0, 42)",
                "llama_model_load: error loading model: llama_model_loader: failed to load model from 03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf",
                "common_fit_params: encountered an error while trying to fit params to free device memory: failed to load model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = probe.build_probe(
        model_source_path=MODEL_SOURCE,
        loader_log_path=loader_log,
        receipt_path=tmp_path / "bonsai_trainability_probe.json",
    )

    assert receipt["schema"] == "lucidota.bonsai.trainability_probe.v1"
    assert receipt["status"] == "BLOCKED"
    assert receipt["exact_target"]["repo"] == "prism-ml/Ternary-Bonsai-8B-gguf"
    assert receipt["exact_target"]["selected_file"] == "Ternary-Bonsai-8B-Q2_0.gguf"
    assert receipt["exact_target"]["matched"] is True
    assert receipt["blockers"] == ["training_unsupported_loader_type_42"]
    assert receipt["loader_failure"]["failure_mode"] == "invalid_ggml_type_42"
    assert receipt["loader_failure"]["loader_command"]
    assert "Ternary-Bonsai-8B-Q2_0.gguf" in " ".join(receipt["loader_failure"]["loader_command"])
    assert receipt["model_calls_performed"] is False
    assert receipt["canonical_graph_writes_performed"] is False
    assert receipt["db_writes_performed"] is False
    assert Path(receipt["receipt_path"]).exists()


def test_probe_rejects_non_exact_target_without_calling_it_trainable(tmp_path: Path) -> None:
    probe = load_probe_module()
    wrong_source = tmp_path / "model_source.json"
    wrong_source.write_text(
        json.dumps(
            {
                "schema": "lucidota.model_source.v1",
                "model_id": "prism-ml/Ternary-Bonsai-8B-gguf",
                "selected_file": "Ternary-Bonsai-8B-Q4_0.gguf",
                "source_url": "https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf",
                "expected_sha256": "deadbeef",
                "format": "GGUF Q4_0",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    loader_log = tmp_path / "loader.log"
    loader_log.write_text("main: loading model\n", encoding="utf-8")

    receipt = probe.build_probe(
        model_source_path=wrong_source,
        loader_log_path=loader_log,
        receipt_path=tmp_path / "probe.json",
    )

    assert receipt["status"] == "BLOCKED"
    assert receipt["exact_target"]["matched"] is False
    assert "exact_target_mismatch" in receipt["blockers"]
    assert receipt["trainable"] is False
