# DARWIN HAMMER — match 3316, survivor 1
# gen: 5
# parent_a: hybrid_workshare_allocator_hybrid_hybrid_hybrid_m1490_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_ternary_route_m910_s2.py (gen4)
# born: 2026-05-29T23:49:19Z

import math
import random
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
import numpy as np

GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
_POLICY: Dict[str, List[float]] = {}          
_STORE: Dict[str, float] = {g: 0.0 for g in GROUPS}  
_COUNTS: Dict[str, int] = {g: 0 for g in GROUPS}    

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def clamp(x: List[float], lower: float, upper: float) -> List[float]:
    return [min(upper, max(lower, xi)) for xi in x]

def hoeffding_epsilon(num_samples: int, delta: float = 0.05) -> float:
    if num_samples <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))

def signal_to_noise_gap(deterministic_units: float, llm_units: float) -> float:
    if llm_units == 0:
        return 1.0
    return max(0.01, deterministic_units / llm_units)

def _pct(value: float) -> float:
    return round(float(value), 6)

def _init_policy_if_missing(context_id: str) -> None:
    if context_id not in _POLICY:
        _POLICY[context_id] = [1.0 / len(GROUPS)] * len(GROUPS)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def compute_confidence(morphology: np.ndarray) -> float:
    if morphology.ndim != 1:
        raise ValueError("morphology must be a 1‑D vector")
    rng = np.random.default_rng(42)
    weights = rng.normal(loc=0.0, scale=1.0, size=morphology.shape)
    bias = rng.normal()
    raw = np.dot(weights, morphology) + bias
    return float(_sigmoid(np.array([raw]))[0])

BASE_COSTS: Dict[str, float] = {
    "codex": 3.0,
    "groq": 2.5,
    "cohere": 4.0,
    "local_models": 1.5,
}

def update_edge_priors(propensities: List[float], confidence: float) -> List[float]:
    blended = [p * (1 - confidence) + confidence for p in propensities]
    total = sum(blended)
    if total == 0:
        return [1.0 / len(blended)] * len(blended)
    return [b / total for b in blended]

def expected_costs(prior_probs: List[float]) -> List[float]:
    costs = []
    for g, p in zip(GROUPS, prior_probs):
        base = BASE_COSTS.get(g, 1.0)
        costs.append(base * (1.0 - p))
    return costs

def allocate_hybrid_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    context_id: str = "global",
) -> Dict[str, Any]:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    _init_policy_if_missing(context_id)
    propensities = _POLICY[context_id]
    prop_array = np.array(propensities, dtype=float)
    prop_array = prop_array / prop_array.sum()
    propensities = prop_array.tolist()
    _POLICY[context_id] = propensities

    allocation = {
        "total_units": total_units,
        "deterministic_units": deterministic_units,
        "llm_units": llm_units,
        "propensities": dict(zip(groups, propensities)),
    }
    return allocation

def route_via_minimum_cost(
    *,
    propensities: List[float],
    confidence: float,
) -> Tuple[str, float, List[float]]:
    posterior = update_edge_priors(propensities, confidence)
    costs = expected_costs(posterior)
    min_idx = int(np.argmin(costs))
    selected_group = GROUPS[min_idx]
    return selected_group, costs[min_idx], posterior

def hybrid_allocate_and_route(
    *,
    total_units: float,
    deterministic_target_pct: float,
    morphology: np.ndarray,
    context_id: str = "global",
) -> Dict[str, Any]:
    allocation = allocate_hybrid_workshare(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        context_id=context_id,
    )
    confidence = compute_confidence(morphology)
    selected_group, expected_cost, posterior = route_via_minimum_cost(
        propensities=allocation["propensities"][GROUPS],
        confidence=confidence,
    )
    allocation["selected_group"] = selected_group
    allocation["expected_cost"] = expected_cost
    allocation["posterior"] = dict(zip(GROUPS, posterior))
    return allocation