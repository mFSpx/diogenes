# DARWIN HAMMER — match 1197, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hoeffding_tre_m301_s0.py (gen4)
# born: 2026-05-29T23:33:29Z

import sys
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Iterable, Any

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    return np.linalg.norm(a - b)


def rbf_similarity_matrix(
    features: List[FeatureVec],
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Vectorised computation of the Gaussian RBF similarity matrix.

    Returns
    -------
    S : ndarray, shape (n, n)
        S[i, j] = exp(-epsilon^2 * ||x_i - x_j||^2)
    """
    X = np.asarray(features, dtype=float)          # (n, d)
    # Using (a-b)^2 = a^2 + b^2 - 2ab
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
    dists_sq = sq_norms + sq_norms.T - 2.0 * X @ X.T           # (n, n)
    np.clip(dists_sq, 0, None, out=dists_sq)                  # numerical safety
    return np.exp(-epsilon ** 2 * dists_sq)


def tropical_relu(matrix: np.ndarray) -> np.ndarray:
    """
    Tropical “max‑plus” projection using the ReLU (max(x,0)) element‑wise.
    """
    return np.maximum(matrix, 0.0)


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Gini coefficient for a non‑negative 1‑D iterable.
    """
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0 or xs.sum() == 0:
        return 0.0
    if np.any(xs < 0):
        raise ValueError("values must be non‑negative")
    xs_sorted = np.sort(xs)
    n = xs_sorted.size
    cum = np.cumsum(xs_sorted)
    # Gini formula using the Lorenz curve
    gini = (n + 1 - 2 * np.sum(cum) / xs_sorted.sum()) / n
    return float(gini)


def hoeffding_bound(sample: np.ndarray, confidence: float = 0.95) -> float:
    """
    Hoeffding bound for a bounded random variable in [0, 1].
    The function assumes the sample lies in that interval (true for ReLU‑projected
    RBF similarities).
    """
    n = sample.size
    if n == 0:
        return float('inf')
    return math.sqrt(math.log(1.0 / (1.0 - confidence)) / (2.0 * n))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Hash‑based utilities (unchanged semantics, but more robust)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Hybrid algorithm – deeper mathematical integration
# ----------------------------------------------------------------------
def _tropical_power(matrix: np.ndarray, power: int) -> np.ndarray:
    """
    Tropical matrix power using max‑plus algebra:
        A ⊗ B = max_i (A[:, i] + B[i, :])
    The implementation leverages broadcasting for efficiency.
    """
    if power < 1:
        raise ValueError("power must be >= 1")
    result = matrix.copy()
    for _ in range(1, power):
        # max‑plus multiplication
        result = np.max(result[:, :, None] + matrix[None, :, :], axis=1)
    return result


def hybrid_rbf_tropical_hoeffding_regret(
    node_features: List[FeatureVec],
    epsilon: float = 1.0,
    tropical_power: int = 2,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Core hybrid routine.

    1. Build a dense RBF similarity matrix.
    2. Project it into the tropical (max‑plus) semiring and optionally
       take a tropical power to capture higher‑order interactions.
    3. Flatten the resulting score matrix to a 1‑D distribution.
    4. Compute the Gini coefficient of that distribution (inequality measure).
    5. Compute a Hoeffding bound for the same distribution.

    Returns
    -------
    gini : float
        Inequality of tropical scores.
    bound : float
        Hoeffding bound at the requested confidence level.
    """
    # 1. RBF similarity
    S = rbf_similarity_matrix(node_features, epsilon=epsilon)

    # 2. Tropical projection + optional power
    T = tropical_relu(S)
    if tropical_power > 1:
        T = _tropical_power(T, tropical_power)

    # 3. Flatten to a single vector (ignore diagonal if desired)
    scores = T.ravel()

    # 4. Gini coefficient
    gini = gini_coefficient(scores)

    # 5. Hoeffding bound (scores lie in [0,1] after ReLU)
    bound = hoeffding_bound(scores, confidence=confidence)

    return gini, bound


def hybrid_hoeffding_tree_split(
    node_features: List[FeatureVec],
    threshold: float,
    epsilon: float = 1.0,
    tropical_power: int = 1,
) -> List[int]:
    """
    Split nodes based on tropical scores that exceed a threshold.
    The tropical scores are obtained after an optional tropical power
    to incorporate multi‑step similarity propagation.
    """
    S = rbf_similarity_matrix(node_features, epsilon=epsilon)
    T = tropical_relu(S)
    if tropical_power > 1:
        T = _tropical_power(T, tropical_power)

    # Use the row‑wise max as a representative score for each node
    node_scores = T.max(axis=1)
    return [i for i, sc in enumerate(node_scores) if sc > threshold]


def hybrid_regret_weighted_ternary_decision(
    node_features: List[FeatureVec],
    epsilon: float = 1.0,
    tropical_power: int = 1,
    ternary_bins: Tuple[float, float] = (0.33, 0.66),
) -> float:
    """
    Produce a regret‑weighted ternary decision vector.

    Steps
    -----
    1. Compute tropical scores (with optional power).
    2. Reduce each node to a single score (row‑wise max).
    3. Quantise scores into ternary values:
           -1 if score < low,
            0 if low ≤ score ≤ high,
            +1 if score > high.
    4. Compute the Gini coefficient of the absolute ternary distribution.
    5. Weight that Gini by a sigmoid‑scaled version of the original scores
       to obtain a regret measure.
    """
    S = rbf_similarity_matrix(node_features, epsilon=epsilon)
    T = tropical_relu(S)
    if tropical_power > 1:
        T = _tropical_power(T, tropical_power)

    node_scores = T.max(axis=1)

    low, high = ternary_bins
    ternary = np.where(
        node_scores < low, -1,
        np.where(node_scores > high, 1, 0)
    )

    # Gini on the magnitude (0 or 1) captures imbalance of decisive vs neutral nodes
    gini = gini_coefficient(np.abs(ternary))

    # Regret weighting
    regret = gini * sigmoid(node_scores).mean()
    return float(regret)


# ----------------------------------------------------------------------
# Demo / sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_features = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]

    gini_val, hoeffding_val = hybrid_rbf_tropical_hoeffding_regret(demo_features)
    print(f"Gini coefficient: {gini_val:.6f}, Hoeffding bound: {hoeffding_val:.6f}")

    split = hybrid_hoeffding_tree_split(demo_features, threshold=0.5)
    print(f"Split nodes (threshold=0.5): {split}")

    regret = hybrid_regret_weighted_ternary_decision(demo_features)
    print(f"Regret‑weighted ternary decision: {regret:.6f}")