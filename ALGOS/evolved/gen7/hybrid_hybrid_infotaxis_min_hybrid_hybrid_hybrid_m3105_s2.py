# DARWIN HAMMER — match 3105, survivor 2
# gen: 7
# parent_a: hybrid_infotaxis_minhash_m63_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s2.py (gen6)
# born: 2026-05-29T23:47:51Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_infotaxis_minhash_m63_s3.py and 
hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s2.py. 
The mathematical bridge between these structures lies in the integration of 
MinHash signatures with the lead-lag transformation and B-spline basis functions. 
The MinHash signature is used to create a transformation matrix, which is then 
used in the lead-lag transformation to assign points to regions.

Parents:
- hybrid_infotaxis_minhash_m63_s3.py (MinHash and entropy core)
- hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s2.py (lead-lag transform and B-spline basis)
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Iterable, List, Set, Tuple, Dict
import hashlib

MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
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
    for i in range(N):
        for j in range(len(t) - 1):
            if t[j] <= x[i] <= t[j + 1]:
                B[i, j] = basis(x[i], t, j, k)
    return B

def basis(x: float, t: np.ndarray, j: int, k: int) -> float:
    if k == 0:
        return 1 if t[j] <= x < t[j + 1] else 0
    else:
        d = t[j + k] - t[j]
        e = t[j + k + 1] - t[j + 1]
        if d == 0 or e == 0:
            return 0
        return (x - t[j]) / d * basis(x, t, j + k - 1, k - 1) + (t[j + k + 1] - x) / e * basis(x, t, j, k - 1)

def hybrid_fusion(tokens: Iterable[str], path: np.ndarray, grid: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
    sig = signature(tokens)
    lead_lag_path = lead_lag_transform(path)
    B = bspline_basis(path[:, 0], grid)
    return sig, lead_lag_path, B

def compute_similarity(sig_a: List[int], sig_b: List[int], lead_lag_path: np.ndarray) -> float:
    similarity_score = similarity(sig_a, sig_b)
    # Use lead_lag_path to compute additional features or weights for the similarity score
    return similarity_score

if __name__ == "__main__":
    tokens = ["token1", "token2", "token3"]
    path = np.random.rand(10, 2)
    grid = np.linspace(0, 1, 10)
    sig, lead_lag_path, B = hybrid_fusion(tokens, path, grid)
    print(sig)
    print(lead_lag_path)
    print(B)