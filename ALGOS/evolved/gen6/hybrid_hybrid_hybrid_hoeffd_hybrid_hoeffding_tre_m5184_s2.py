# DARWIN HAMMER — match 5184, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m1988_s1.py (gen5)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s5.py (gen1)
# born: 2026-05-30T00:00:26Z

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0
ALPHA: float = 5.0
LAMBDA: float = 0.7
MINHASH_K: int = 64
MAX64: int = (1 << 64) - 1
SEED: int = 12345

random.seed(SEED)

# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra utilities
# ----------------------------------------------------------------------
def tropical_max(*values: float) -> float:
    """Tropical addition (max) over an arbitrary number of scalars."""
    if not values:
        raise ValueError("tropical_max requires at least one argument")
    return max(values)


def tropical_plus(a: float, b: float) -> float:
    """Tropical multiplication (ordinary addition)."""
    return a + b


def tropical_weighted_max(gains: List[float], weights: List[float]) -> float:
    """
    Compute the tropical weighted vote:
        max_i (gain_i ⊗ weight_i)   where ⊗ is ordinary multiplication.
    """
    if len(gains) != len(weights):
        raise ValueError("gains and weights must have the same length")
    weighted = [g * w for g, w in zip(gains, weights)]
    return tropical_max(*weighted)


# ----------------------------------------------------------------------
# Hoeffding bound utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Classic Hoeffding bound for a range‑bounded random variable.
    Returns ε such that P(|X - E[X]| > ε) ≤ δ.
    """
    if r <= 0:
        raise ValueError("Range r must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("Delta must satisfy 0 < delta < 1")
    if n <= 0:
        raise ValueError("Sample size n must be positive")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Decision data structure
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    metric: float
    reason: str


# ----------------------------------------------------------------------
# Core hybrid decision rule
# ----------------------------------------------------------------------
def _normalize_weights(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights cannot be zero")
    return [w / total for w in weights]


def _tropical_gap_metric(gap: float, eps: float, tie_thresh: float) -> Tuple[bool, str, float]:
    """
    Combine gap and ε in the tropical semiring.
    We compute ρ = max(gap - ε, -tie_thresh).
    - If ρ > 0 → split (gap decisively larger than bound)
    - If ρ == 0 → tie, split if ε is already below the tie threshold
    - Otherwise → wait
    """
    # Tropical addition (max) of the two terms
    metric = tropical_max(gap - eps, -tie_thresh)
    if metric > 0:
        return True, "gap_exceeds_bound", metric
    if eps < tie_thresh:
        return True, "epsilon_below_tie", metric
    return False, "wait", metric


def hybrid_hoeffding_tropical(
    gains: List[float],
    weights: List[float],
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """
    Hybrid Hoeffding‑tropical max‑plus decision rule.

    Steps
    -----
    1. Normalise the supplied weights.
    2. Compute all tropical weighted gains.
    3. Identify the best and second‑best weighted gains.
    4. Evaluate the Hoeffding bound ε.
    5. Use a tropical combination of (gap‑ε) and the tie threshold
       to decide whether to split.
    """
    # 1. Normalise weights
    norm_weights = _normalize_weights(weights)

    # 2. Tropical weighted gains for each attribute
    weighted_gains = [g * w for g, w in zip(gains, norm_weights)]

    # 3. Best and second‑best (handle ties robustly)
    sorted_vals = sorted(weighted_gains, reverse=True)
    best_gain = sorted_vals[0]
    second_best_gain = sorted_vals[1] if len(sorted_vals) > 1 else float("-inf")
    gain_gap = best_gain - second_best_gain

    # 4. Hoeffding bound
    eps = hoeffding_bound(r, delta, n)

    # 5. Tropical decision metric
    split, reason, metric = _tropical_gap_metric(gain_gap, eps, tie_threshold)

    return SplitDecision(
        should_split=split,
        epsilon=eps,
        gain_gap=gain_gap,
        metric=metric,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example_gains = [0.52, 0.31, 0.18, 0.12]
    example_weights = [0.4, 0.3, 0.2, 0.1]
    r_val = 0.5
    delta_val = 0.1
    n_samples = 120

    decision = hybrid_hoeffding_tropical(
        gains=example_gains,
        weights=example_weights,
        r=r_val,
        delta=delta_val,
        n=n_samples,
        tie_threshold=0.04,
    )
    print(decision)