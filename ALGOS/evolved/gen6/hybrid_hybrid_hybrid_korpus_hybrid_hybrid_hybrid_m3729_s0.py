# DARWIN HAMMER — match 3729, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_korpus_text_h_hybrid_hybrid_hybrid_m2438_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1321_s1.py (gen5)
# born: 2026-05-29T23:51:19Z

"""
Hybrid algorithm merging:
- `hybrid_hybrid_korpus_text_h_hybrid_hybrid_hybrid_m2438_s0.py` (MinHash-based similarity and regret-weighted strategy)
- `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1321_s1.py` (B-spline basis functions and weekday weight vector)

The mathematical bridge between these two algorithms lies in the use of geometric sphericity to define a characteristic angle θ, 
and then evaluating the B-spline basis functions at this angle to obtain a relevance weight w_b.
This weight is then used to modulate the regret-weighting term in the first algorithm.

"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    return [_hash(i, t) for i, t in enumerate(sorted(toks))[:k]]

def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    intersection = set(sig1) & set(sig2)
    union = set(sig1) | set(sig2)
    return len(intersection) / len(union)

def fisher_information(theta: float) -> float:
    return 1 / (theta ** 2)

def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def bspline_basis(x, grid, k=3, weights=None):
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    if weights is not None:
        B *= weights

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term = (t[i + order] - x) / denom_r
            for j in range(order):
                B_new[:, i + j] += B[:, i + j + 1] * term / denom_l

    return B

def hybrid_operation(sig1: List[int], sig2: List[int], groups: list[str], dow: int) -> float:
    similarity = jaccard_similarity(sig1, sig2)
    theta = math.acos(similarity)
    weights = weekday_weight_vector(groups, dow)
    grid = np.linspace(0, 2 * math.pi, len(groups))
    B = bspline_basis(theta, grid, weights=weights)
    relevance_weight = np.sum(B * weights)
    return fisher_information(theta) * relevance_weight

def test_hybrid_operation():
    sig1 = signature(["apple", "banana", "orange"])
    sig2 = signature(["banana", "orange", "grape"])
    groups = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = 3  # Thursday
    result = hybrid_operation(sig1, sig2, groups, dow)
    print(result)

if __name__ == "__main__":
    test_hybrid_operation()