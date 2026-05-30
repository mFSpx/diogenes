# DARWIN HAMMER — match 3349, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_gini_coeffici_hybrid_hybrid_worksh_m2277_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s0.py (gen3)
# born: 2026-05-29T23:49:32Z

"""
This module fuses the mathematical structures of hybrid_hybrid_gini_coeffici_hybrid_hybrid_worksh_m2277_s1.py 
and hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s0.py. The bridge between the two parents lies 
in their use of similarity matrices and matrix operations. Specifically, Parent A's gini_coefficient 
function uses a sorted list of values to calculate the Gini coefficient, while Parent B's 
weekday_weight_vector function uses a sinusoidal rotation to generate a row-stochastic vector. 
The hybrid algorithm combines these two concepts by using the Gini coefficient to modulate the 
weekday-derived weight vector with the similarity information.

The mathematical interface between the two parents can be expressed as:

gini_coef = gini_coefficient(similarities)
weight_vec = 1.0 + amplitude * np.sin(base_angles + phase)
modulated_weight_vec = weight_vec * gini_coef
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Hashable, Sequence, List, Tuple, Iterable, Any

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – Gini and similarity utilities
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a non‑negative value collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 1 bit per value (up to 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values:
        bits = (bits << 1) | (1 if v > avg else 0)
    return bits

def similarity_matrix(groups: Sequence[FeatureVec]) -> np.ndarray:
    """Compute similarity matrix for groups using perceptual hashes."""
    n = len(groups)
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarities[i, j] = 1 - abs(compute_phash(groups[i]) ^ compute_phash(groups[j])) / 64
    return similarities

# ----------------------------------------------------------------------
# Parent B – weekday weight vector
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7)
    return (1.0 + np.sin(base_angles + phase)) / n

def modulate_weight_vector(weight_vec: np.ndarray, gini_coef: float) -> np.ndarray:
    """Modulate weight vector with Gini coefficient."""
    return weight_vec * gini_coef

def hybrid_allocation(groups: Sequence[FeatureVec], dow: int) -> np.ndarray:
    """Compute hybrid allocation for groups using similarity matrix and weekday weight vector."""
    similarities = similarity_matrix(groups)
    gini_coef = gini_coefficient(np.mean(similarities, axis=1))
    weight_vec = weekday_weight_vector([str(i) for i in range(len(groups))], dow)
    modulated_weight_vec = modulate_weight_vector(weight_vec, gini_coef)
    return modulated_weight_vec

if __name__ == "__main__":
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    dow = 3
    print(hybrid_allocation(groups, dow))