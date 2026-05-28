"""Deterministic Project 2501 workshare math.

Pure allocator: scripts own receipts/DB/queue writes. No model calls.
"""
from __future__ import annotations

from typing import Any

GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    return round(float(value), 6)


def allocate_workshare(*, total_units: float, deterministic_target_pct: float = 90.0, groups: tuple[str, ...] = GROUPS) -> dict[str, Any]:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]
    jzloads: list[dict[str, Any]] = [
        {
            "kind": "OBJECT",
            "id": "project2501_workshare_policy",
            "type": "workshare_policy",
            "deterministic_target_pct": _pct(deterministic_target_pct),
            "llm_residual_pct": _pct(100.0 - deterministic_target_pct),
        },
        {
            "kind": "EVENT",
            "id": "project2501_workshare_allocation",
            "type": "allocation_computed",
            "total_units": _pct(total_units),
            "deterministic_units": _pct(deterministic_units),
            "llm_units": _pct(llm_units),
        },
    ]
    for lane in lanes:
        jzloads.append(
            {
                "kind": "EDGE",
                "from": "project2501_workshare_policy",
                "to": f"model_group:{lane['group']}",
                "type": "ASSIGNS_LLM_RESIDUAL_QUARTER",
                "llm_units": lane["llm_units"],
                "llm_share_pct": lane["llm_share_pct"],
            }
        )
    return {
        "schema": "lucidota.project2501.workshare_allocation.v1",
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
        "jzloads": jzloads,
    }


def summarize_savings(*, total_units: float, deterministic_target_pct: float = 90.0) -> dict[str, Any]:
    plan = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    return {
        "schema": "lucidota.project2501.token_savings.v1",
        "baseline_llm_units": _pct(total_units),
        "planned_llm_units": plan["llm_units"],
        "deterministic_units": plan["deterministic_units"],
        "token_savings_pct": _pct((total_units - plan["llm_units"]) / total_units * 100.0),
        "per_group_llm_units": {lane["group"]: lane["llm_units"] for lane in plan["lanes"]},
    }
