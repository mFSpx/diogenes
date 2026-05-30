# DARWIN HAMMER — match 3061, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m914_s0.py (gen5)
# parent_b: hybrid_hybrid_shap_attribut_hybrid_hybrid_hybrid_m1561_s0.py (gen5)
# born: 2026-05-29T23:47:35Z

"""Hybrid Regret‑Shapley‑VRAM Scheduler
Integrates:
- Parent A: Tropical max‑plus leader election, Hoeffding bound, reconstruction risk, VRAM scheduling.
- Parent B: Shapley value attribution, lead‑lag transform, hash‑based distances, broadcast probability.

Mathematical bridge:
The bridge is the *value function* used inside the Shapley kernel.  In Parent A the
value of a set of actions is given by a tropical max‑plus polynomial
`max_i (c_i + g)`.  We reuse this tropical evaluation as the cooperative game
value for the Shapley computation of Parent B.  Consequently the Shapley
attribution inherits the max‑plus algebraic structure, while the lead‑lag
transform injects temporal dependencies into the gain vector that feeds the
tropical polynomial.  The resulting Shapley scores are then weighted by the
reconstruction‑risk score and finally used to drive a VRAM‑aware model‑pool
scheduler.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Node = str
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Parent‑A structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Simple descriptor of a model tier."""
    name: str
    ram_mb: int
    tier: str


def compute_hoeffding_bound(observed_gains: List[float],
                            epsilon: float,
                            confidence: float) -> float:
    """Hoeffding bound for a list of observed gains."""
    if not observed_gains:
        return float('inf')
    return math.sqrt(2 * math.log(1 / confidence) / len(observed_gains))


def tropical_max_plus_evaluate(coefficients: List[float],
                               gain: float) -> float:
    """Evaluate a tropical max‑plus polynomial: max_i (c_i + gain)."""
    if not coefficients:
        return -math.inf
    return max(c + gain for c in coefficients)


def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Risk score in [0,1] based on quasi‑identifier uniqueness."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


# ----------------------------------------------------------------------
# Parent‑B structures
# ----------------------------------------------------------------------
def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Standard Shapley kernel weight."""
    return (math.factorial(subset_size) *
            math.factorial(feature_count - subset_size - 1) /
            math.factorial(feature_count))


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a 2‑D path (time × dimension).
    Returns an interleaved sequence of forward and backward increments.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        # forward increment
        out[2 * t] = np.concatenate([path[t + 1] - path[t], np.zeros(d)])
        # backward increment
        out[2 * t + 1] = np.concatenate([np.zeros(d), path[t + 1] - path[t]])
    out[-1] = np.concatenate([np.zeros(d), path[-1] - path[-2]])
    return out


# ----------------------------------------------------------------------
# Hybrid core functions (the required three+ functions)
# ----------------------------------------------------------------------
def tropical_shapley_value(feature_idx: int,
                           feature_count: int,
                           coefficients: List[float],
                           gains: List[float]) -> float:
    """
    Shapley value where the cooperative game value is a tropical max‑plus
    evaluation.  For any subset S of features the value is

        v(S) = max_{i∈S} (c_i + Σ_{j∈S} g_j)

    where c_i are `coefficients` and g_j are `gains`.
    """
    def value_fn(S: frozenset[int]) -> float:
        if not S:
            return -math.inf
        subset_coeffs = [coefficients[i] for i in S]
        subset_gain = sum(gains[i] for i in S)
        return tropical_max_plus_evaluate(subset_coeffs, subset_gain)

    total = 0.0
    for k in range(feature_count + 1):
        weight = shapley_kernel_weight(k, feature_count)
        for subset in combinations(range(feature_count), k):
            S = frozenset(subset)
            marginal = value_fn(S | {feature_idx}) - value_fn(S)
            total += weight * marginal
    return total


def lead_lag_tropical_schedule(path_gains: np.ndarray,
                               model_tiers: List[ModelTier],
                               ram_ceiling_mb: int) -> List[ModelTier]:
    """
    1. Apply the lead‑lag transform to a temporal gain matrix.
    2. For each transformed step compute a tropical max‑plus score using the
       RAM size of each model tier as the coefficient vector.
    3. Greedily select models in descending score order while respecting the
       VRAM ceiling.
    Returns the selected list of `ModelTier` objects.
    """
    transformed = lead_lag_transform(path_gains)  # shape (2T‑1, 2d)
    # Collapse the 2d dimension to a single scalar gain per step (L2 norm)
    step_gains = np.linalg.norm(transformed, axis=1)

    # Prepare coefficient vectors: each tier contributes its RAM as a coefficient
    coeff_matrix = np.array([tier.ram_mb for tier in model_tiers], dtype=float)

    # Compute tropical scores per tier (max over steps)
    tier_scores = []
    for idx, tier in enumerate(model_tiers):
        scores = [tropical_max_plus_evaluate([coeff_matrix[idx]],
                                             gain) for gain in step_gains]
        tier_scores.append((tier, max(scores)))

    # Sort by descending tropical score
    tier_scores.sort(key=lambda x: x[1], reverse=True)

    selected: List[ModelTier] = []
    used_ram = 0
    for tier, _score in tier_scores:
        if used_ram + tier.ram_mb <= ram_ceiling_mb:
            selected.append(tier)
            used_ram += tier.ram_mb
    return selected


def hybrid_regret_vram_shap_scheduler(actions: List[Any],
                                      counterfactuals: List[Any],
                                      model_tiers: List[ModelTier],
                                      gains: List[float],
                                      ram_ceiling_mb: int,
                                      uqis: int,
                                      total_records: int) -> List[ModelTier]:
    """
    End‑to‑end hybrid scheduler:

    * Compute a reconstruction‑risk score (Parent A).
    * Build a synthetic temporal path from `gains` and apply lead‑lag transform.
    * Derive tropical‑Shapley attributions for each model tier.
    * Weight those attributions by the risk score.
    * Perform VRAM‑aware selection.
    Returns the final list of scheduled `ModelTier`s.
    """
    # 1. Risk weighting
    risk = reconstruction_risk_score(uqis, total_records)

    # 2. Temporal path – treat gains as a 1‑D trajectory and embed in 2‑D
    #    (gain, gain^2) to give the transform something to work with.
    gains_arr = np.asarray(gains, dtype=float)
    path = np.column_stack([gains_arr, gains_arr ** 2])

    # 3. Lead‑lag + tropical scores per tier
    transformed = lead_lag_transform(path)
    step_gains = np.linalg.norm(transformed, axis=1)

    # 4. Compute tropical‑Shapley attribution for each tier
    shapley_scores = []
    coeffs = [tier.ram_mb for tier in model_tiers]  # use RAM as tropical coeffs
    for idx, tier in enumerate(model_tiers):
        # Use the tropical_shapley_value with the tier's own coefficient
        shap = tropical_shapley_value(
            feature_idx=idx,
            feature_count=len(model_tiers),
            coefficients=coeffs,
            gains=step_gains.tolist()
        )
        # Weight by risk and by the average tropical evaluation over steps
        tropical_avg = np.mean([tropical_max_plus_evaluate([coeffs[idx]], g)
                                for g in step_gains])
        weighted = shap * tropical_avg * (1.0 - risk)
        shapley_scores.append((tier, weighted))

    # 5. Greedy VRAM selection based on the hybrid score
    shapley_scores.sort(key=lambda x: x[1], reverse=True)
    selected: List[ModelTier] = []
    used_ram = 0
    for tier, _score in shapley_scores:
        if used_ram + tier.ram_mb <= ram_ceiling_mb:
            selected.append(tier)
            used_ram += tier.ram_mb
    return selected


# ----------------------------------------------------------------------
# Utility for combinations (required by shapley computation)
# ----------------------------------------------------------------------
def combinations(iterable: Sequence[int], r: int):
    """Yield combinations of `iterable` taken `r` at a time (lightweight)."""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny model pool
    tiers = [
        ModelTier(name="tiny", ram_mb=256, tier="A"),
        ModelTier(name="small", ram_mb=512, tier="B"),
        ModelTier(name="medium", ram_mb=1024, tier="C"),
        ModelTier(name="large", ram_mb=2048, tier="D")
    ]

    # Synthetic gains (e.g., validation improvements over epochs)
    gains = [random.uniform(-0.5, 1.5) for _ in range(8)]

    # Run the three core functions
    print("=== Tropical‑Shapley demo ===")
    coeffs = [tier.ram_mb for tier in tiers]
    for i in range(len(tiers)):
        val = tropical_shapley_value(i, len(tiers), coeffs, gains)
        print(f"Feature {i} (tier {tiers[i].name}) Shapley ≈ {val:.4f}")

    print("\n=== Lead‑lag tropical schedule ===")
    path = np.column_stack([gains, np.square(gains)])
    selected = lead_lag_tropical_schedule(path, tiers, ram_ceiling_mb=3000)
    print("Selected tiers (VRAM ≤ 3000 MB):", [t.name for t in selected])

    print("\n=== End‑to‑end hybrid scheduler ===")
    final_selection = hybrid_regret_vram_shap_scheduler(
        actions=[],
        counterfactuals=[],
        model_tiers=tiers,
        gains=gains,
        ram_ceiling_mb=3000,
        uqis=42,
        total_records=1000
    )
    print("Final scheduled tiers:", [t.name for t in final_selection])