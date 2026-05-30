# DARWIN HAMMER — match 3316, survivor 0
# gen: 5
# parent_a: hybrid_workshare_allocator_hybrid_hybrid_hybrid_m1490_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_ternary_route_m910_s2.py (gen4)
# born: 2026-05-29T23:49:19Z

"""Hybrid Workshare Allocation & Bayesian Minimum‑Cost Routing

Parents:
- **Parent A** (`hybrid_workshare_allocator_hybrid_hybrid_hybrid_m1490_s5.py`):
  Provides a workshare allocator that maintains per‑group propensities (a
  probability vector) and distributes deterministic vs. LLM units.

- **Parent B** (`hybrid_hybrid_hybrid_label__hybrid_ternary_route_m910_s2.py`):
  Supplies a KAN‑style confidence estimator for a morphology vector and a
  ternary router that selects an engine channel by minimizing the Bayesian‑
  updated expected cost  
  `C(g) = Σ base_cost(g)·(1‑p_g)` where `p_g` are edge priors.

Mathematical Bridge
-------------------
Both parents manipulate probability‑like quantities:

* Parent A’s **propensities** `π_g` are a normalized distribution over groups.
* Parent B’s **confidence** `c∈[0,1]` can be interpreted as a prior that the
  current request belongs to the “correct” group.

The fusion treats the confidence `c` as a *global evidence* that updates the
propensities `π_g` into *edge priors* `p_g` via a simple Bayesian‑style
averaging:


p_g ← (π_g + c) / 2


The updated priors are then used in Parent B’s cost formula to compute an
expected cost for each group.  The group with the minimal expected cost is
chosen as the routing destination, and the workshare allocator finally
splits the total units between deterministic work and LLM work according to
the selected group’s share.

The module below implements this unified system, exposing three core
functions that demonstrate the hybrid operation and a smoke‑test that runs
end‑to‑end without external dependencies.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and simple utilities (from Parent A)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
_POLICY: Dict[str, List[float]] = {}          # context → propensities per group
_STORE: Dict[str, float] = {g: 0.0 for g in GROUPS}  # virtual store per group
_COUNTS: Dict[str, int] = {g: 0 for g in GROUPS}    # observations per group


def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule used by the original allocator."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: List[float], lower: float, upper: float) -> List[float]:
    """Clamp each element of a list to the interval [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


def hoeffding_epsilon(num_samples: int, delta: float = 0.05) -> float:
    """Hoeffding bound epsilon for a given sample count."""
    if num_samples <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))


def signal_to_noise_gap(deterministic_units: float, llm_units: float) -> float:
    """Ratio of deterministic to LLM units, lower‑bounded."""
    if llm_units == 0:
        return 1.0
    return max(0.01, deterministic_units / llm_units)


def _pct(value: float) -> float:
    return round(float(value), 6)


def _init_policy_if_missing(context_id: str) -> None:
    """Lazy‑initialise a uniform propensity vector for a new context."""
    if context_id not in _POLICY:
        _POLICY[context_id] = [1.0 / len(GROUPS)] * len(GROUPS)


# ----------------------------------------------------------------------
# Parent B – confidence estimator (tiny KAN surrogate) and routing tree
# ----------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_confidence(morphology: np.ndarray) -> float:
    """
    Very small KAN‑style surrogate: a single linear layer followed by a sigmoid.
    Returns a confidence score `c ∈ [0,1]`.
    """
    if morphology.ndim != 1:
        raise ValueError("morphology must be a 1‑D vector")
    # Random but reproducible weights for the demo
    rng = np.random.default_rng(42)
    weights = rng.normal(loc=0.0, scale=1.0, size=morphology.shape)
    bias = rng.normal()
    raw = np.dot(weights, morphology) + bias
    return float(_sigmoid(np.array([raw]))[0])


# Routing tree is a flat mapping group → base cost (could be extended to a real tree)
BASE_COSTS: Dict[str, float] = {
    "codex": 3.0,
    "groq": 2.5,
    "cohere": 4.0,
    "local_models": 1.5,
}


def update_edge_priors(propensities: List[float], confidence: float) -> List[float]:
    """
    Bayesian‑style averaging of the existing propensity vector with the global
    confidence evidence `c`.  The result stays normalized.
    """
    blended = [(p + confidence) / 2.0 for p in propensities]
    total = sum(blended)
    if total == 0:
        # fallback to uniform
        return [1.0 / len(blended)] * len(blended)
    return [b / total for b in blended]


def expected_costs(prior_probs: List[float]) -> List[float]:
    """
    Compute expected cost for each group using the formula
        C(g) = base_cost(g) * (1 - p_g)
    where `p_g` is the updated prior probability for group `g`.
    Returns a list aligned with `GROUPS`.
    """
    costs = []
    for g, p in zip(GROUPS, prior_probs):
        base = BASE_COSTS.get(g, 1.0)
        costs.append(base * (1.0 - p))
    return costs


# ----------------------------------------------------------------------
# Hybrid core: allocate workshare and route based on confidence
# ----------------------------------------------------------------------
def allocate_hybrid_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    context_id: str = "global",
) -> Dict[str, Any]:
    """
    Mirrors Parent A’s allocator but also returns the normalized propensity
    vector for the given context.  The function validates inputs, normalises
    the stored propensities and splits the total units into deterministic and
    LLM portions.
    """
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
    """
    Implements Parent B’s minimum‑cost routing using the confidence‑updated
    priors.  Returns the selected group, its expected cost, and the full
    posterior probability vector.
    """
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
    """
    End‑to‑end hybrid operation:

    1. Allocate deterministic vs. LLM workshare (Parent A).
    2. Compute a confidence score from the morphology vector (Parent B).
    3. Update group propensities with the confidence and compute expected costs.
    4. Choose the cheapest group and report the final allocation.

    The returned dictionary contains all intermediate artefacts for inspection.
    """
    # Step 1 – allocation and fetch current propensities
    alloc = allocate_hybrid_workshare(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        context_id=context_id,
    )
    prop_vec = [alloc["propensities"][g] for g in GROUPS]

    # Step 2 – confidence from morphology
    confidence = compute_confidence(morphology)

    # Step 3 & 4 – routing
    chosen_group, exp_cost, posterior = route_via_minimum_cost(
        propensities=prop_vec,
        confidence=confidence,
    )

    # Record the choice back into the policy (optional learning step)
    _POLICY[context_id] = posterior

    result = {
        "allocation": alloc,
        "confidence": _pct(confidence),
        "chosen_group": chosen_group,
        "expected_cost": _pct(exp_cost),
        "posterior_propensities": dict(zip(GROUPS, posterior)),
    }
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic reproducibility for the demo
    random.seed(0)
    np.random.seed(0)

    # Example parameters
    TOTAL_UNITS = 120.0
    DET_TARGET_PCT = 75.0
    CONTEXT = "demo"

    # Random morphology vector (e.g., extracted features)
    morph_vec = np.random.rand(8)

    output = hybrid_allocate_and_route(
        total_units=TOTAL_UNITS,
        deterministic_target_pct=DET_TARGET_PCT,
        morphology=morph_vec,
        context_id=CONTEXT,
    )

    print("Hybrid Allocation & Routing Result")
    print("---------------------------------")
    for key, value in output.items():
        print(f"{key}: {value}")