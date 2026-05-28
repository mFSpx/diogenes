from core.telemetry.sanity_gate import TelemetrySanityGate, safe_numeric_metrics_from_text


def test_anti_ghost_telemetry_mirroring():
    mock_raw_nvidia_smi = "NVIDIA GeForce GTX 1650, 57 C, 2850 MiB, 1246 MiB"
    bad_parsed_output = {
        "gpu_temp_c": 57,
        "vram_used_mb": 1650,
        "vram_free_mb": 2065,
    }

    gate = TelemetrySanityGate(known_static_tokens=["1650"])
    audit = gate.verify_metric_integrity(mock_raw_nvidia_smi, bad_parsed_output)

    assert audit["status"] == "DIAGNOSTIC_WARNING_METRIC_MIRROR_DETECTED"
    assert "vram_used_mb" in audit["collisions_detected"]
    assert audit["corrected_metrics"]["vram_used_mb"] is None


def test_safe_numeric_metrics_pass_through_when_memory_fields_are_separate():
    raw = "memory.used=1650, memory.free=2065"
    parsed = {"vram_used_mb": 1650, "vram_free_mb": 2065}
    cleaned = safe_numeric_metrics_from_text(raw, parsed, known_static_tokens=["1650"])
    assert cleaned["vram_used_mb"] == 1650
    assert cleaned["vram_free_mb"] == 2065


def test_safe_numeric_metrics_leave_unrelated_numbers_alone():
    raw = "memory.used=1066, memory.free=2065"
    parsed = {"vram_used_mb": 1066, "vram_free_mb": 2065}
    cleaned = safe_numeric_metrics_from_text(raw, parsed, known_static_tokens=["1650"])
    assert cleaned["vram_used_mb"] == 1066
    assert cleaned["vram_free_mb"] == 2065
