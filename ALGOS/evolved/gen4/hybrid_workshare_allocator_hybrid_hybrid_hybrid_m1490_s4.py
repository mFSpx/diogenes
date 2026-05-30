# DARWIN HAMMER — match 1490, survivor 4
# gen: 4
# parent_a: workshare_allocator.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s2.py (gen3)
# born: 2026-05-29T23:37:01Z

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants and shared structures
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
_POLICY: Dict[str, List[float]] = {}          # context → propensities per group
_STORE: Dict[str, float] = {g: 0.0 for g in GROUPS}  # virtual store per group
_COUNTS: Dict[str, int] = {g: 0 for g in GROUPS}    # observations per group

# ----------------------------------------------------------------------
# Utility helpers (from Parent B)
# ----------------------------------------------------------------------
def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def clamp(x: List[float], lower: float, upper: float) -> List[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]

def hoeffding_epsilon(num_samples: int, delta: float = 0.05) -> float:
    """Hoeffding bound ε = sqrt(ln(2/δ) / (2·n))."""
    if num_samples <= 0:
        return 1.0  # maximal uncertainty when no data
    return math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))

def signal_to_noise_gap(deterministic_units: float, llm_units: float) -> float:
    """Simple S/N gap: ratio of deterministic to LLM units, bounded >0."""
    if llm_units == 0:
        return 1.0
    return max(0.01, deterministic_units / llm_units)

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round to six decimal places for reproducibility."""
    return round(float(value), 6)

def _init_policy_if_missing(context_id: str) -> None:
    """Lazy initialise uniform propensities for a new context."""
    if context_id not in _POLICY:
        _POLICY[context_id] = [1.0 / len(GROUPS)] * len(GROUPS)

def allocate_hybrid_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    context_id: str = "global",
) -> Dict[str, Any]:
    """
    Allocate ``total_units`` according to the hybrid scheme.

    Returns a dictionary mirroring Parent A's output but with adaptive LLM
    shares derived from the bandit policy.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    # Initialise / fetch bandit propensities for the given context
    _init_policy_if_missing(context_id)
    propensities = _POLICY[context_id]
    prop_array = np.array(propensities, dtype=float)
    # Normalise to a probability simplex (guard against numerical drift)
    prop_array = prop_array / prop_array.sum()
    propensities = prop_array.tolist()
    _POLICY[context_id] = propensities  # store back the normalised version

    # Allocate LLM residual proportionally to propensities
    shares = [p * llm_units for p in propensities]

    # Update virtual store using the store equation
    sn_gap = signal_to_noise_gap(deterministic_units, llm_units)
    for idx, group in enumerate(groups):
        inflow = shares[idx]
        # Outflow is a decaying function of current store (eviction)
        outflow = evasion_delta(_COUNTS[group], t_max=100) * _STORE[group]
        eps = hoeffding_epsilon(_COUNTS[group])
        new_store = _STORE[group] + inflow - outflow * sn_gap * eps
        _STORE[group] = max(0.0, new_store)  # store cannot be negative

    # Build lane structures compatible with Parent A
    lanes = [
        {
            "group": group,
            "llm_units": _pct(shares[i]),
            "llm_share_pct": _pct(100.0 * shares[i] / llm_units) if llm_units else 0.0,
            "proof_required": True,
            "store": _pct(_STORE[group]),
        }
        for i, group in enumerate(groups)
    ]

    # Assemble jzloads (metadata) – kept simple but illustrative
    jzloads: List[Dict[str, Any]] = [
        {
            "kind": "OBJECT",
            "id": "project2501_hybrid_workshare_policy",
            "type": "workshare_policy",
            "deterministic_target_pct": _pct(deterministic_target_pct),
            "llm_residual_pct": _pct(100.0 - deterministic_target_pct),
        },
        {
            "kind": "EVENT",
            "id": "project2501_hybrid_allocation",
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
                "from": "project2501_hybrid_workshare_policy",
                "to": f"model_group:{lane['group']}",
                "type": "ASSIGNS_LLM_RESIDUAL_DYNAMIC",
                "llm_units": lane["llm_units"],
                "llm_share_pct": lane["llm_share_pct"],
                "store": lane["store"],
            }
        )

    return {
        "schema": "lucidota.project2501.hybrid_workshare_allocation.v1",
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
        "jzloads": jzloads,
    }

def record_reward(
    *,
    context_id: str,
    group: str,
    reward: float,
    propensity: float | None = None,
) -> None:
    """
    Update the bandit policy for ``group`` within ``context_id`` using the
    observed ``reward``. If ``propensity`` is provided, use it to scale the
    learning rate.
    """
    _COUNTS[group] += 1
    if context_id not in _POLICY:
        raise ValueError("context_id not found")

    propensities = _POLICY[context_id]
    epsilon = hoeffding_epsilon(_COUNTS[group])
    learning_rate = epsilon

    if propensity is not None:
        learning_rate *= propensity

    # Update bandit propensities
    idx = GROUPS.index(group)
    propensities[idx] = max(0.0, propensities[idx] + learning_rate * reward)
    propensities = clamp(propensities, 0.0, 1.0)
    propensities = [p / sum(propensities) for p in propensities]
    _POLICY[context_id] = propensities

def simulate_step(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    context_id: str = "global",
) -> Dict[str, Any]:
    """
    Simulate one step of the hybrid allocation.

    Returns the updated allocation and bandit policy.
    """
    allocation = allocate_hybrid_workshare(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        context_id=context_id,
    )

    # Simulate rewards for each group
    rewards = [random.random() for _ in GROUPS]

    for idx, group in enumerate(GROUPS):
        record_reward(
            context_id=context_id,
            group=group,
            reward=rewards[idx],
        )

    return allocation