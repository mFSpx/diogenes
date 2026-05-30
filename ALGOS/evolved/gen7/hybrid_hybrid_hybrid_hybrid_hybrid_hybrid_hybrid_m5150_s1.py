# DARWIN HAMMER — match 5150, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ssim_h_hybrid_hybrid_hybrid_m2479_s1.py (gen6)
# born: 2026-05-30T00:00:04Z

"""
Hybrid algorithm merging Regret-weighted strategy (Parent A) with 
geometric-multivector similarity and virtual-store bandit (Parent B).

The mathematical bridge between the two parents lies in the use of 
regret-weighted probabilities as a proxy for the confidence bounds 
in the bandit's expected reward calculation. Specifically, we 
replace the uniform exploration term in Parent B's propensity 
calculation with the regret-weighted probabilities from Parent A.

This allows the hybrid algorithm to leverage both the regret-based 
strategy and the geometric-multivector similarity in a unified 
framework.

Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s2.py
Parent B: hybrid_hybrid_hybrid_ssim_h_hybrid_hybrid_hybrid_m2479_s1.py
"""

import os
import sys
import math
import random
import pathlib
import datetime
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# VRAM-aware learning-rate utilities
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if pathlib.Path(__file__).exists() else pathlib.Path.cwd()
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def _now_iso_z() -> str:
    """Current UTC timestamp in ISO-8601 with trailing “Z”."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def _mock_free_vram_mb() -> int:
    """Return a pseudo-random free-VRAM estimate (for CPU-only testing)."""
    total = 8192
    used = random.randint(0, total - DEFAULT_RESERVE_MB)
    return max(total - used - DEFAULT_RESERVE_MB, 0)

def budgeted_lr(base_lr: float,
                free_mb: int,
                budget_mb: int = DEFAULT_BUDGET_MB,
                reserve_mb: int = DEFAULT_RESERVE_MB) -> float:
    """
    Scale ``base_lr`` according to available VRAM.

    If the free memory exceeds the usable budget (budget – reserve) the full
    learning-rate is returned; otherwise a linear decay down to 10 % is applied.
    """
    usable = max(budget_mb - reserve_mb, 1)
    if free_mb >= usable:
        return base_lr
    scale = 0.1 + 0.9 * (free_mb / usable)
    return base_lr * scale

# ----------------------------------------------------------------------
# Regret-weighted strategy
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(regrets: List[float]) -> List[float]:
    """
    Convert a list of non-negative regrets into a probability distribution.

    The classic regret-matching rule is used:
        p_i ∝ max(regret_i, 0)
    The probabilities sum to one; if all regrets are zero a uniform distribution
    is returned.
    """
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    if total == 0.0:
        n = len(regrets)
        return [1.0 / n] * n
    return [p / total for p in positive]

# ----------------------------------------------------------------------
# Multivector and geometric similarity
# ----------------------------------------------------------------------
class Multivector:
    """Simple Euclidean Clifford algebra up to grade 2."""

    def __init__(self, components: dict, n: int):
        # Remove near‑zero components for cleanliness
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)  # dimension of the underlying vector space

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            result[blade] = result.get(blade, 0.0) + value
        return Multivector(result, self.n)

def stats_to_multivector(stats: List[float]) -> Multivector:
    """Build a grade‑0/1/2 multivector from the first two statistical moments."""
    mean = np.mean(stats)
    var = np.var(stats)
    return Multivector({frozenset(): mean, frozenset({0}): var}, 1)

def geometric_ssim(mv1: Multivector, mv2: Multivector) -> float:
    """Return the scalar part of the geometric product of two multivectors."""
    return mv1.scalar_part() * mv2.scalar_part() + sum(mv1.components.get(b, 0.0) * mv2.components.get(b, 0.0) for b in set(mv1.components) & set(mv2.components))

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_algorithm(regrets: List[float], stats: List[List[float]], 
                     alpha: float, beta: float, delta_t: float) -> Tuple[List[float], Multivector]:
    """
    Run the hybrid algorithm.

    Parameters:
    - regrets: list of non-negative regrets
    - stats: list of lists of statistical moments
    - alpha, beta: parameters for the virtual-store dynamics
    - delta_t: time step

    Returns:
    - propensity: list of probabilities
    - store: updated multivector
    """
    # Compute regret-weighted probabilities
    propensity = compute_regret_weighted_strategy(regrets)

    # Initialize store
    store = Multivector({frozenset(): 0.0}, 1)

    # Iterate over stats
    for stat in stats:
        # Build multivector
        mv = stats_to_multivector(stat)

        # Compute similarity
        sim = geometric_ssim(mv, store)

        # Update store
        store = Multivector({frozenset(): store.scalar_part() + delta_t * (alpha * sim - beta * store.scalar_part())}, 1)

        # Compute expected rewards
        expected_rewards = [sim + store.scalar_part() for _ in range(len(regrets))]

        # Compute new propensity
        new_propensity = [p * (1 - propensity[i]) + (1 - p) * propensity[i] * expected_rewards[i] for i, p in enumerate(propensity)]

    return new_propensity, store

def main():
    regrets = [1.0, 2.0, 3.0]
    stats = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    alpha = 0.1
    beta = 0.2
    delta_t = 0.01

    propensity, store = hybrid_algorithm(regrets, stats, alpha, beta, delta_t)
    print(propensity, store.scalar_part())

if __name__ == "__main__":
    main()