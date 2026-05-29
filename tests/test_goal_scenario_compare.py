from scripts import goal_scenario_compare


def test_compare_scenario_reports_identifies_rule_deltas_and_next_seed():
    baseline = {
        "schema": "lucidota.goal_scenario_batch.v1",
        "decision_rule_candidates": [
            {"condition": {"family": "normal", "intent": "ops", "lane": "SLOWLANE", "hygiene_label": "neutral_or_unclear", "model_needed": False}, "action": {"outbound_state": "draft_only"}, "support": 4},
            {"condition": {"family": "fast_slow", "intent": "ops", "lane": "SLOWLANE", "hygiene_label": "neutral_or_unclear", "model_needed": False}, "action": {"outbound_state": "draft_only"}, "support": 2},
        ],
    }
    current = {
        "schema": "lucidota.goal_scenario_holdout.v1",
        "promoted_rules": [
            {"condition": {"family": "normal", "intent": "ops", "lane": "SLOWLANE", "hygiene_label": "neutral_or_unclear", "model_needed": False}, "action": {"outbound_state": "draft_only"}, "support": 5},
            {"condition": {"family": "fast_slow", "intent": "ops", "lane": "SLOWLANE", "hygiene_label": "neutral_or_unclear", "model_needed": False}, "action": {"outbound_state": "final_print"}, "support": 3},
            {"condition": {"family": "ontology_pressure", "intent": "ontology", "lane": "SLOWLANE", "hygiene_label": "neutral_or_unclear", "model_needed": False}, "action": {"outbound_state": "draft_only"}, "support": 3},
        ],
    }

    report = goal_scenario_compare.compare_reports(current, baseline)

    assert report["schema"] == "lucidota.goal_scenario_compare.v1"
    assert report["ontology_mode"] == "GO25_STRICT"
    assert len(report["stable_rules"]) == 1
    assert len(report["morphing_rules"]) == 1
    assert len(report["new_rules"]) == 1
    assert len(report["lost_rules"]) == 0
    assert report["next_seed"] == ["OBJECT", "EVENT", "EDGE"]
    assert "fast_slow" in report["scenario_focus"]
