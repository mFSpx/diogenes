from scripts.needle_shared_prefix_runtime import build_exact_encoder_reuse_plan


def test_exact_encoder_reuse_plan_collapses_identical_full_encoder_inputs():
    lanes = [{"query": "same 500 token chunk", "tools": [{"name": "go25"}]} for _ in range(6)]
    plan = build_exact_encoder_reuse_plan(lanes)
    assert plan["lane_count"] == 6
    assert plan["unique_encoder_inputs"] == 1
    assert plan["encode_calls_saved"] == 5
    assert plan["exact_full_encoder_input_reuse"] is True
    assert plan["can_share_for_all_lanes"] is True
    assert plan["unique_inverse"] == [0, 0, 0, 0, 0, 0]


def test_exact_encoder_reuse_plan_refuses_prefix_reuse_when_tools_differ():
    lanes = [
        {"query": "same 500 token chunk", "tools": [{"name": "go25"}]},
        {"query": "same 500 token chunk", "tools": [{"name": "gcio75"}]},
    ]
    plan = build_exact_encoder_reuse_plan(lanes)
    assert plan["lane_count"] == 2
    assert plan["unique_encoder_inputs"] == 2
    assert plan["encode_calls_saved"] == 0
    assert plan["exact_full_encoder_input_reuse"] is False
    assert plan["can_share_for_all_lanes"] is False
    assert "different lane tools/tasks" in plan["boundary"]
