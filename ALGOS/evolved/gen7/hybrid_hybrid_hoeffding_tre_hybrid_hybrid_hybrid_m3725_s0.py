# DARWIN HAMMER — match 3725, survivor 0
# gen: 7
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s6.py (gen1)
# parent_b: hybrid_hybrid_hybrid_tropic_hybrid_hybrid_krampu_m2522_s0.py (gen6)
# born: 2026-05-29T23:51:18Z

"""
Hybrid Algorithm: Hoeffding-Gini Tree Fusion with Tropical Feature Binding

This module integrates two parent algorithms:
- hybrid_hoeffding_tree_gini_coefficient_m13_s6.py, which provides a hybrid Hoeffding-Gini module for decision tree growth.
- hybrid_hybrid_tropic_hybrid_hybrid_krampu_m2522_s0.py, which implements a tropical-feature fusion with fractional power binding.

The mathematical bridge lies in using the tropical evaluation to drive the analog fractional binding in the Hoeffding-Gini tree growth. 
The Gini gain is used as the tropical coefficient, and the Hoeffding bound is applied to the gain gap. 
The tropical polynomial evaluation is then used to determine the split decision.

"""

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

# ----------------------------------------------------------------------
# Parent A – Hoeffding helpers
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε for range *r*, confidence *δ* and sample count *n*."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑based split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


# ----------------------------------------------------------------------
# Parent B – Tropical algebra primitives
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition (max). Broadcasts like NumPy ufunc."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (ordinary addition). Broadcasts."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical matrix multiplication.

    (A ⊗ B)[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # shape (m, p, 1) + (1, p, n) -> (m, p, n) then max over p
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def t_polyval(coeffs, x):
    """
    Evaluate a tropical polynomial at x.

    p(x) = max_i ( coeffs[i] + i * x )
    """
    coeffs = np.asarray(coeffs, dtype=float)
    return np.max(coeffs + np.arange(len(coeffs)) * x)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def gini_impurity(class_freqs: List[float]) -> float:
    """Compute the Gini impurity of a class-frequency vector."""
    total = sum(class_freqs)
    if total == 0:
        return 0.0
    return 1.0 - sum((freq / total) ** 2 for freq in class_freqs)


def gini_gain(best_gain: float, second_best_gain: float, r: float, delta: float, n: int) -> SplitDecision:
    """Hybrid split decision using Gini gain and Hoeffding bound."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    if gap > eps:
        return SplitDecision(True, eps, gap, "Gini gain exceeds Hoeffding bound")
    else:
        return SplitDecision(False, eps, gap, "Gini gain does not exceed Hoeffding bound")


def hybrid_split_decision(coeffs: List[float], x: float, r: float, delta: float, n: int) -> SplitDecision:
    """Hybrid split decision using tropical polynomial evaluation and Hoeffding bound."""
    gain = t_polyval(coeffs, x)
    second_best_gain = t_polyval(coeffs, x - 1)
    return gini_gain(gain, second_best_gain, r, delta, n)


def demonstrate_hybrid_operation():
    """Demonstrate the hybrid operation."""
    coeffs = [1.0, 2.0, 3.0]
    x = 2.0
    r = 1.0
    delta = 0.05
    n = 100
    decision = hybrid_split_decision(coeffs, x, r, delta, n)
    print(f"Should split: {decision.should_split}, Epsilon: {decision.epsilon}, Gain gap: {decision.gain_gap}, Reason: {decision.reason}")


if __name__ == "__main__":
    demonstrate_hybrid_operation()