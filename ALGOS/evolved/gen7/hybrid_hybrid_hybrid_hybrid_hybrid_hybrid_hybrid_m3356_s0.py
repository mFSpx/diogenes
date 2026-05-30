# DARWIN HAMMER — match 3356, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_label__m1580_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tri_al_m1845_s2.py (gen5)
# born: 2026-05-29T23:49:24Z

"""
Hybrid Algorithm Fusion of Clifford Algebra and Bayesian Labeling

Parents:
- Parent A: `hybrid_hybrid_hybrid_geomet_hybrid_hybrid_model__m176_s1.py` – provides
  Clifford (geometric) algebra operations on multivectors.
- Parent B: `hybrid_hybrid_tri_algo_cond_hybrid_hard_truth_ma_m755_s0.py` – provides
  Lead-lag transform and B-spline basis utilities.

Mathematical Bridge:
The scalar (grade-0) part of the geometric product of two vectors in a Clifford
algebra equals their Euclidean dot product. We exploit this property to turn the
Clifford product into a *context-sensitive similarity* between feature vectors.
That similarity then modulates the Bayesian update performed by Parent B,
yielding a hybrid confidence that is aware of geometric context while retaining
the probabilistic rigor of the original labeling framework.

The mathematical interface between the parents is established through the
following mapping:
- Clifford algebra's geometric product (A, B) -> <A, B> = A · B
- Lead-lag transform's interleaving of channels (A, B) -> interleaved(A, B)
- B-spline basis's evaluation of functions at positions (A, B) -> B(A, B)

This fusion combines the geometric context of Clifford algebra with the probabilistic
rigor of Bayesian labeling, creating a hybrid system that is aware of both spatial
relationships and uncertainty.
"""

import sys
import math
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# ----------------------------------------------------------------------
# Immutable decision container
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConduitDecision:
    """Immutable container for a decision made by the hybrid system."""
    action: str
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str


# ----------------------------------------------------------------------
# Clifford algebra core (modified to interface with Bayesian labeling)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel the pair and restart scanning from the beginning
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                n -= 2
                i = -1  # will become 0 after i += 1 at loop bottom
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return sign * np.prod([2**i for i in result])

def geometric_product(a, b):
    """Compute the geometric product of two multivectors."""
    return np.sum([_multiply_blades(blade_a, blade_b) for blade_a in a for blade_b in b])


# ----------------------------------------------------------------------
# Bayesian labeling core (modified to interface with Clifford algebra)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Interleave lead-lag channels.
    Input: (T, d) array.
    Output: (2T-1, 2d) array.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2-D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B-spline basis functions of order k at positions x.
    Returns B where B[i, j] = B_j(x_i).
    The number of basis functions equals len(grid) + k - 1.
    """
    grid = np.asarray(grid, dtype=float)
    if k < 1:
        raise ValueError("Spline order k must be >= 1")
    # Clamp ends with multiplicity k
    left = np.full(k, grid[0])
    right = np.full(k, grid[-1])
    augmented_grid = np.concatenate([left, grid, right])
    B = np.zeros((len(x), len(augmented_grid) + k - 1))
    for i, x_i in enumerate(x):
        for j, t in enumerate(augmented_grid):
            if t <= x_i < t + 1:
                B[i, j] = (x_i - t) ** (k - 1) / math.factorial(k - 1)
            elif x_i == t:
                B[i, j] = 1
    return B

def hybrid_confidence(a, b):
    """Compute the hybrid confidence of two multivectors."""
    dot_product = np.dot(a, b)
    lead_lag_interleaved = lead_lag_transform(b)
    bspline_basis_evaluated = bspline_basis(lead_lag_interleaved, a)
    return dot_product * np.sum(bspline_basis_evaluated)


# ----------------------------------------------------------------------
# Test the hybrid system
# ----------------------------------------------------------------------
if __name__ == "__main__":
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(hybrid_confidence(a, b))