# DARWIN HAMMER — match 4532, survivor 1
# gen: 4
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s9.py (gen1)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py (gen3)
# born: 2026-05-29T23:56:22Z

"""
This module fuses the governing equations of hybrid_hoeffding_tree_gini_coefficient_m13_s9.py and 
hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py. 
The mathematical bridge between their structures lies in the use of Hoeffding bounds 
in both algorithms for determining split decisions. 
The fusion integrates the Gini coefficient calculations from the first parent 
with the RBF kernel matrix computations from the second parent.

Parent A: hybrid_hoeffding_tree_gini_coefficient_m13_s9.py
Parent B: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py
"""

import math
import random
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) ).

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

@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑Gini split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def gini_impurity_from_counts(counts: Counter) -> float:
    """Gini impurity given a Counter of class frequencies."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return 1.0 - np.sum(probs ** 2)

def gini_gain(parent_counts: Counter,
              left_counts: Counter,
              right_counts: Counter) -> float:
    """Reduction in Gini impurity obtained by splitting ``parent`` into left/right.

    This version works directly with Counters to avoid materialising label lists.
    """
    n_parent = sum(parent_counts.values())
    if n_parent == 0:
        return 0.0

    parent_imp = gini_impurity_from_counts(parent_counts)
    left_imp = gini_impurity_from_counts(left_counts)
    right_imp = gini_impurity_from_counts(right_counts)

    n_left = sum(left_counts.values())
    n_right = sum(right_counts.values())

    weighted_imp = (n_left / n_parent) * left_imp + (n_right / n_parent) * right_imp
    return parent_imp - weighted_imp

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rbf_kernel_matrix(features: Dict[int, List[float]], epsilon: float = 1.0) -> np.ndarray:
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K

def hybrid_hoeffding_gini_rbf(features: Dict[int, List[float]], 
                              parent_counts: Counter, 
                              left_counts: Counter, 
                              right_counts: Counter, 
                              range_: float, delta: float, n: int) -> Tuple[SplitDecision, np.ndarray]:
    gini_red = gini_gain(parent_counts, left_counts, right_counts)
    eps = hoeffding_bound(range_, delta, n)
    should_split = gini_red > eps
    gain_gap = gini_red - eps
    reason = f"Gini reduction {gini_red:.4f} > Hoeffding bound {eps:.4f}"
    split_decision = SplitDecision(should_split, eps, gain_gap, reason)
    rbf_kernel = rbf_kernel_matrix(features)
    return split_decision, rbf_kernel

def demonstrate_hybrid_operation():
    features = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0], 2: [7.0, 8.0, 9.0]}
    parent_counts = Counter({0: 10, 1: 20})
    left_counts = Counter({0: 5, 1: 10})
    right_counts = Counter({0: 5, 1: 10})
    range_ = 10.0
    delta = 0.01
    n = 100
    split_decision, rbf_kernel = hybrid_hoeffding_gini_rbf(features, parent_counts, left_counts, right_counts, range_, delta, n)
    print(split_decision)
    print(rbf_kernel)

if __name__ == "__main__":
    demonstrate_hybrid_operation()