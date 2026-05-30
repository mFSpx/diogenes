# DARWIN HAMMER — match 4532, survivor 0
# gen: 4
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s9.py (gen1)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py (gen3)
# born: 2026-05-29T23:56:22Z

"""
This module implements a hybrid algorithm that fuses the topologies of the 
PARENT ALGORITHM A (hybrid_hoeffding_tree_gini_coefficient_m13_s9.py) and 
PARENT ALGORITHM B (hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py).

The mathematical bridge between their structures is formed by combining the 
Hoeffding bound from both parents with the Gini impurity from PARENT ALGORITHM A 
and the RBF kernel matrix from PARENT ALGORITHM B. This allows for a hybrid 
split decision to be made based on both the Hoeffding bound and the similarity 
between feature vectors.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple
import numpy as np
from collections import Counter

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """
    Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) ).

    Args:
        range_: The known range R of the bounded random variable (max - min).
        delta: Desired error probability (0 < δ < 1).
        n: Number of independent observations.

    Returns:
        The confidence radius ε.
    """
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("range_ > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))

def gini_impurity_from_counts(counts: Counter) -> float:
    """
    Gini impurity given a Counter of class frequencies.

    Args:
        counts: A Counter of class frequencies.

    Returns:
        The Gini impurity.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return 1.0 - np.sum(probs ** 2)

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """
    Gaussian function.

    Args:
        r: The input value.
        epsilon: The epsilon value (default is 1.0).

    Returns:
        The result of the Gaussian function.
    """
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """
    Euclidean distance between two vectors.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The Euclidean distance.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """
    Compute the p-hash value.

    Args:
        values: The list of values.

    Returns:
        The p-hash value.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """
    Hamming distance between two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The Hamming distance.
    """
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute the similarity matrix.

    Args:
        features: A dictionary of node to feature vector.

    Returns:
        The similarity matrix and the list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[n])) for n in nodes]
    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def rbf_kernel_matrix(features: Dict[Node, FeatureVec], epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute the RBF kernel matrix.

    Args:
        features: A dictionary of node to feature vector.
        epsilon: The epsilon value (default is 1.0).

    Returns:
        The RBF kernel matrix and the list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes

@dataclass(frozen=True)
class SplitDecision:
    """
    Result of a Hoeffding-Gini split test.
    """
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    """
    Determine whether to split based on the Hoeffding bound and Gini impurity.

    Args:
        best_gain: The best gain.
        second_best_gain: The second best gain.
        r: The range of the bounded random variable.
        delta: The desired error probability.
        n: The number of independent observations.
        tie_threshold: The tie threshold (default is 0.05).

    Returns:
        The split decision.
    """
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_best_gain
    should_split = gain_gap > eps
    reason = "Hoeffding bound" if should_split else "Gini impurity"
    return SplitDecision(should_split, eps, gain_gap, reason)

def hybrid_split(features: Dict[Node, FeatureVec], counts: Counter, r: float, delta: float, n: int) -> SplitDecision:
    """
    Hybrid split decision based on the Hoeffding bound, Gini impurity, and RBF kernel matrix.

    Args:
        features: A dictionary of node to feature vector.
        counts: A Counter of class frequencies.
        r: The range of the bounded random variable.
        delta: The desired error probability.
        n: The number of independent observations.

    Returns:
        The split decision.
    """
    S, nodes = similarity_matrix(features)
    K, _ = rbf_kernel_matrix(features)
    gini_imp = gini_impurity_from_counts(counts)
    best_gain = 0.0
    second_best_gain = 0.0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            gain = gini_imp - (S[i, j] * K[i, j])
            if gain > best_gain:
                second_best_gain = best_gain
                best_gain = gain
            elif gain > second_best_gain:
                second_best_gain = gain
    return should_split(best_gain, second_best_gain, r, delta, n)

if __name__ == "__main__":
    features = {0: [1.0, 2.0], 1: [3.0, 4.0], 2: [5.0, 6.0]}
    counts = Counter({0: 2, 1: 3})
    r = 1.0
    delta = 0.01
    n = 10
    decision = hybrid_split(features, counts, r, delta, n)
    print(decision)