# DARWIN HAMMER — match 3493, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m602_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s5.py (gen4)
# born: 2026-05-29T23:50:25Z

"""
This module fuses the core topologies of hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m602_s2.py (PARENT ALGORITHM A)
and hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s5.py (PARENT ALGORITHM B) into a unified system.

The mathematical interface between the two parents lies in the use of geometric algebra and information-theoretic measures.
PARENT ALGORITHM A uses a perceptual hash (phash) and Gini coefficient, while PARENT ALGORITHM B uses multivectors and Clifford algebra.
The hybrid algorithm combines the phash and Gini coefficient with multivectors to create a more comprehensive system.

The governing equations of PARENT ALGORITHM A are integrated with the matrix operations of PARENT ALGORITHM B through the use of multivectors,
which can be represented as sparse matrices. The phash and Gini coefficient are used to weight the multivector components,
allowing for a more nuanced and flexible representation of complex data.

"""

import numpy as np
import math
import random
from typing import Dict, List, Tuple
from pathlib import Path

def compute_phash(values: List[float]) -> int:
    """
    Compute a 64‑bit perceptual hash of a numeric sequence.
    The hash is based on whether each element is above or below the mean.
    Empty input yields 0.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:                     # truncate / pad to 64 bits
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def gini_coefficient(rewards: List[float]) -> float:
    """
    Compute the Gini coefficient of a reward batch.
    Handles zero‑mean and empty inputs gracefully.
    """
    rewards_arr = np.asarray(rewards, dtype=float)
    if rewards_arr.size == 0:
        return 0.0
    mean = rewards_arr.mean()
    if mean == 0.0:
        return 0.0
    # Vectorised Gini: sort, then use the known formula
    sorted_rewards = np.sort(rewards_arr)
    n = rewards_arr.size
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_rewards)) / (n * np.sum(sorted_rewards)) - (n + 1) / n
    return float(gini)


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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
                del lst[j : j + 2]
                n -= 2
                i = -1  
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = dict(components or {})

    def __add__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
            if abs(res[k]) < 1e-15:
                del res[k]
        return Multivector(res)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        result: Dict[frozenset, float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                coeff = ca * cb * sign
                result[blade] = result.get(blade, 0.0) + coeff
        result = {k: v for k, v in result.items() if abs(v) > 1e-15}
        return Multivector(result)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


def vector_to_mv(x: float, y: float) -> Multivector:
    return Multivector({frozenset({0}): x, frozenset({1}): y})


def hybrid_operation(values: List[float], x: float, y: float) -> Multivector:
    phash_val = compute_phash(values)
    gini_val = gini_coefficient(values)
    mv = vector_to_mv(x, y)
    weighted_mv = Multivector({k: v * gini_val for k, v in mv.components.items()})
    return weighted_mv


def hybrid_distance(mv_a: Multivector, mv_b: Multivector) -> float:
    phash_a = 0
    phash_b = 0
    for k, v in mv_a.components.items():
        phash_a ^= hash(frozenset(k))
    for k, v in mv_b.components.items():
        phash_b ^= hash(frozenset(k))
    return hamming_distance(phash_a, phash_b)


def hybrid_info_gain(mv: Multivector, values: List[float]) -> float:
    gini_val = gini_coefficient(values)
    scalar_part = mv.scalar_part()
    return gini_val * scalar_part


if __name__ == "__main__":
    values = [random.random() for _ in range(100)]
    x = 1.0
    y = 2.0
    mv = hybrid_operation(values, x, y)
    print(mv)
    mv_b = vector_to_mv(3.0, 4.0)
    print(hybrid_distance(mv, mv_b))
    print(hybrid_info_gain(mv, values))