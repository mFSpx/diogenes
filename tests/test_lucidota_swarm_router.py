from scripts import lucidota_swarm_router as router


def test_cost_policy_can_choose_cheapest_capable_model_lane_without_network():
    decision = router.route_task({
        "task_id": "cost_sensitive_public_language_judgment",
        "security_level": "public",
        "privacy_sensitivity": "none",
        "latency_preference": "batch",
        "compute_weight": "light",
        "required_capabilities": ["text", "structured_output"],
        "allow_external": True,
        "prefer_local": False,
        "requires_model_judgment": True,
        "expected_tokens": 2000,
        "metadata": {
            "cost_priority": "high",
            "lane_cost_units_per_1k_tokens": {
                "local_sovereign": 500,
                "groq_like_fast": 10,
                "cohere_like_context": 100,
                "default_airlock": 900,
            },
        },
    })

    assert decision["decision"]["primary_lane"] == "groq_like_fast"
    assert decision["routing_policy"]["cost_aware"] is True
    assert decision["cost_evaluation"]["priority"] == "high"
    assert decision["cost_evaluation"]["lanes"]["groq_like_fast"]["source"] == "task_metadata_override"
    assert decision["sovereignty_guards"]["provider_calls"] is False
    assert decision["sovereignty_guards"]["implicit_network"] is False


def test_cost_policy_never_pulls_exact_receipts_out_of_deterministic_lane():
    decision = router.route_task({
        "task_id": "cost_sensitive_exact_receipt",
        "security_level": "sovereign",
        "privacy_sensitivity": "sensitive",
        "latency_preference": "urgent",
        "compute_weight": "tiny",
        "required_capabilities": ["policy", "exact_check", "receipt"],
        "allow_external": True,
        "prefer_local": False,
        "requires_model_judgment": False,
        "expected_tokens": 200,
        "metadata": {
            "cost_priority": "high",
            "lane_cost_units_per_1k_tokens": {
                "deterministic_workflow": 0,
                "groq_like_fast": 1,
            },
        },
    })

    assert decision["decision"]["primary_lane"] == "deterministic_workflow"
    assert decision["decision"]["model_lane"] is False
    assert decision["sovereignty_guards"]["llm_for_deterministic_work"] is False
