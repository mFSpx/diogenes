from scripts.board_effect_doctrine import evaluate_board_effect, extract_board_effects, select_board_persona


def test_krampus_requires_two_or_more_board_effects():
    result = evaluate_board_effect(
        text="audit one receipt only",
        persona="krampus",
        evidence_refs=["file://receipt.json"],
        explicit_effects=["receipt"],
    )
    assert result["verdict"] == "FAIL"
    assert "krampus_requires_at_least_two_board_effects" in result["blockers"]


def test_krampus_passes_multi_effect_move_with_evidence():
    result = evaluate_board_effect(
        text="audit krampuschewing, emit receipts, update graph stage, train routing rows",
        persona="krampus",
        evidence_refs=["file://05_OUTPUTS/x.json"],
    )
    assert result["verdict"] == "PASS"
    assert result["effect_count"] >= 4
    assert result["fairness"] == "evidence_bound"


def test_krampus_cannot_invent_naughty_without_evidence():
    result = evaluate_board_effect(
        text="mark this script naughty slop and quarantine it",
        persona="krampus",
        evidence_refs=[],
    )
    assert result["verdict"] == "FAIL"
    assert "naughty_claim_without_evidence" in result["blockers"]


def test_santa_settles_for_one_board_effect():
    result = evaluate_board_effect(
        text="find new glow in this receipt",
        persona="santa",
        evidence_refs=[],
        explicit_effects=["glow"],
    )
    assert result["verdict"] == "PASS"
    assert result["minimum_effect_count"] == 1
    assert result["assumption_policy"] == "may_seek_glow_but_must_label_assumptions"


def test_project2501_board_move_carries_effect_gate():
    from scripts.project2501_board_move import build_board_move

    bundle = build_board_move(
        actor="operator",
        source="operator_chat",
        text="Krampus audit krampuschewing, emit receipts, stage graph candidates, refresh model routing rows.",
        execute=True,
        position="krampuschewing_graph_recovery",
    )
    gate = bundle["board_move"]["effect_gate"]
    assert gate["persona"] == "krampus"
    assert gate["effect_count"] >= 4
    assert gate["verdict"] == "PASS"


def test_persona_selector_uses_krampus_for_audit_slop_graph_pressure():
    assert select_board_persona("audit slop and corpse manifest") == "krampus"
    assert select_board_persona("find glow in a clean receipt") == "santa"


def test_effect_extraction_is_deterministic_and_deduped():
    effects = extract_board_effects("graph graph receipts model routing tests")
    assert effects == sorted(set(effects))
    assert {"graph", "receipt", "model", "routing", "test"}.issubset(set(effects))
