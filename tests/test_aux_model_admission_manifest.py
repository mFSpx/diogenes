import json
import subprocess
import sys
from pathlib import Path

MANIFEST = Path("04_RUNTIME/aux_model_admission_manifest.json")


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_exists_and_forces_refs_not_bodies_contract():
    data = load_manifest()
    event_ref = data["event_ref_contract"]
    assert event_ref["mpsc_moves"] == "refs_not_bodies"
    assert "String" in event_ref["forbidden_event_body_types"]
    assert event_ref["max_preview_bytes"] <= 512
    for key in [
        "max_event_bytes",
        "max_stdout_bytes",
        "max_stderr_bytes",
        "max_prompt_bytes",
        "max_json_bytes",
        "max_response_tokens",
        "max_audio_buffer_ms",
        "max_db_rows",
        "max_file_read_bytes",
    ]:
        assert data["global_budgets"][key] > 0


def test_aux_models_are_not_default_residents_and_have_cages():
    data = load_manifest()
    aux = [t for t in data["tools"] if t["class"] in {"CPU_WARM_ONE_AT_A_TIME", "COLD_SUBPROCESS", "BATCH_OFFLINE_ONLY"}]
    assert {t["id"] for t in aux} >= {
        "embedder_onnx_cpu",
        "reranker_cpu",
        "ocr_tesseract",
        "ocr_paddle_cold",
        "gliner_local",
        "whisper_audio",
        "piper_tts",
        "vision_ocr_layout",
    }
    for tool in aux:
        assert tool["resident_default"] is False
        assert tool["receipt_required"] is True
        assert tool["limits"]["max_input_bytes"] > 0
        assert tool["limits"]["max_output_bytes"] > 0
        assert tool["limits"]["timeout_ms"] > 0
        assert tool["cgroup"]["MemoryHighMB"] > 0
        assert tool["cgroup"]["MemoryMaxMB"] >= tool["cgroup"]["MemoryHighMB"]


def test_ssd_deep_and_aux_burst_are_mutually_exclusive():
    data = load_manifest()
    mutexes = {tuple(sorted(m)) for m in data["mutual_exclusion"]}
    assert tuple(sorted(("SSD_DEEP", "AUX_MODEL_BURST"))) in mutexes
    assert data["governor_rules"]["memory_over_pct"]["action"] == "stop_ssd_model"
    assert data["governor_rules"]["vram_over_pct"]["action"] == "drop_second_bonsai_slot"


def test_cli_validates_manifest_and_admits_small_embedder():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/aux_model_admission.py",
            "--manifest",
            str(MANIFEST),
            "--tool",
            "embedder_onnx_cpu",
            "--input-bytes",
            "2048",
            "--memory-pct",
            "40",
            "--vram-pct",
            "30",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["admit"] is True
    assert result["tool"] == "embedder_onnx_cpu"
    assert result["mode"] == "EVIDENCE_MODE"


def test_cli_rejects_aux_when_ssd_deep_active():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/aux_model_admission.py",
            "--manifest",
            str(MANIFEST),
            "--tool",
            "ocr_paddle_cold",
            "--input-bytes",
            "2048",
            "--active-lane",
            "SSD_DEEP",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 2
    result = json.loads(proc.stdout)
    assert result["admit"] is False
    assert "mutual_exclusion" in result["reason"]
