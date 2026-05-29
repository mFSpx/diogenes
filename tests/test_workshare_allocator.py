from __future__ import annotations

from ALGOS.workshare_allocator import allocate_workshare, summarize_savings


def test_workshare_allocator_enforces_even_model_quarters_and_deterministic_target() -> None:
    plan = allocate_workshare(total_units=100, deterministic_target_pct=90)
    assert plan["deterministic_units"] == 90
    assert plan["llm_units"] == 10
    assert {lane["group"]: lane["llm_units"] for lane in plan["lanes"]} == {
        "codex": 2.5,
        "groq": 2.5,
        "cohere": 2.5,
        "local_models": 2.5,
    }
    assert plan["jzloads"][0]["kind"] == "OBJECT"
    assert plan["jzloads"][1]["kind"] == "EVENT"
    assert all(edge["kind"] == "EDGE" for edge in plan["jzloads"][2:])


def test_workshare_savings_compares_all_llm_baseline_to_90_percent_algos() -> None:
    summary = summarize_savings(total_units=1000, deterministic_target_pct=90)
    assert summary["baseline_llm_units"] == 1000
    assert summary["planned_llm_units"] == 100
    assert summary["token_savings_pct"] == 90.0
    assert summary["per_group_llm_units"] == {
        "codex": 25.0,
        "groq": 25.0,
        "cohere": 25.0,
        "local_models": 25.0,
    }
