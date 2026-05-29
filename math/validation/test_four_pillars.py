from ALGOS.percyphon import procedural_entity_generator
from core.language_membrane import route_inbound_text, weave_output
from core.telemetry.diogenes import compress_activity, staple_activity


def test_diogenes_compresses_mouse_and_key_activity():
    packet = compress_activity(
        {"mouse_deltas": [{"dx": 3, "dy": 4}, {"dx": 1, "dy": -2}], "keystroke_burst": 7, "click_count": 2},
        {"scroll_count": 1},
        window_seconds=2,
    )
    assert packet["mouse_delta_sum"] == 10.0
    assert packet["keystroke_burst"] == 7
    assert packet["flow_friction_score"] == 5.0


def test_diogenes_staples_packet():
    packet = staple_activity({"source": "absurd"}, {"mouse_delta_sum": 5, "keystroke_burst": 2})
    assert packet["compressed_activity"]["mouse_delta_sum"] == 5.0
    assert packet["keystroke_burst"] == 2


def test_percyphon_generates_zero_vram_slots_with_offsets():
    result = procedural_entity_generator(
        villagers=[f"villager-{i}" for i in range(32)],
        psyche_wrath_velocity=0.9,
        psyche_forensic_shield_ratio=0.1,
        fluid_slots=8,
    )
    assert result["zero_vram"] is True
    assert result["slot_count"] == 12
    assert result["ternary_offset"] == 1
    assert len(result["slots"]) == 12
    assert len(result["fluid_slots"]) == 8


def test_language_membrane_regex_and_weave():
    routed = route_inbound_text("FOR UPDATE SKIP LOCKED is active in the pipeline")
    assert routed["lane"] == "rete_regex"
    woven = weave_output(
        deterministic_template="draft_only manifest",
        rag_quotes=[{"doc_id": "x", "quote": "quote"}],
        deepseek_synthesis="synthesis",
        fairyfuse_context={"source": "math"},
    )
    assert woven["outbound_state"] == "draft_only"
    assert woven["smoothing"]["backend"].startswith("FAIRYFUSE")
