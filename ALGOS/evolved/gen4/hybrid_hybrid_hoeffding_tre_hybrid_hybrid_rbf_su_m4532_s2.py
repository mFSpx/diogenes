# DARWIN HAMMER — match 4532, survivor 2
# gen: 4
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s9.py (gen1)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py (gen3)
# born: 2026-05-29T23:56:22Z

"""Hybrid Hoeffding‑Gini‑RBF module.

Parents:
- **hybrid_hoeffding_tree_gini_coefficient_m13_s9.py** – provides a Hoeffding bound
  and Gini‑impurity based split gain.
- **hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py** – provides an
  RBF‑kernel similarity matrix (Gaussian kernel) and a second implementation of
  the Hoeffding bound.

Mathematical bridge:
Both parents rely on a confidence radius `ε = sqrt(R²·ln(1/δ)/(2·n))`.  The first
parent uses it to decide whether a *Gini gain* is statistically significant.
The second parent produces a similarity measure `S(i,j)=exp(-(‖x_i‑x_j‖·ε)²)`,
which is bounded in `[0,1]`.  By treating `1‑S̄(L,R)` (average dissimilarity
between left/right partitions) as an additional impurity‑reduction term we can
form a **hybrid gain**


G_hybrid = α·G_gini + β·(1‑S̄(L,R)),   α,β ≥ 0, α+β = 1


The same Hoeffding bound then tests the statistical significance of this
combined gain, yielding a unified split decision that respects both label
purity and feature‑space separation.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Sequence, Hashable, Set

import numpy as np

# ----------------------------------------------------------------------
# Core utilities shared by both parents
# ----------------------------------------------------------------------
def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding confidence radius ε = sqrt( (R²·ln(1/δ)) / (2·n) )."""
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("range_>0, 0<delta<1, n>0 required")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Gini‑impurity utilities (Parent A)
# ----------------------------------------------------------------------
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
    """Reduction in Gini impurity obtained by splitting ``parent`` into left/right."""
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


# ----------------------------------------------------------------------
# RBF‑kernel utilities (Parent B)
# ----------------------------------------------------------------------
def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian (RBF) kernel value for distance ``r``."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_kernel_matrix(features: Dict[Hashable, Sequence[float]],
                     epsilon: float = 1.0) -> Tuple[np.ndarray, List[Hashable]]:
    """Full symmetric RBF kernel matrix K[i,j] = exp(-ε²·‖x_i‑x_j‖²)."""
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


def average_rbf_dissimilarity(left_ids: List[Hashable],
                              right_ids: List[Hashable],
                              kernel: np.ndarray,
                              node_index: Dict[Hashable, int]) -> float:
    """
    Compute 1‑average similarity between two partitions.
    Returns a value in [0,1] where 0 means perfectly similar,
    1 means maximally dissimilar.
    """
    if not left_ids or not right_ids:
        return 0.0
    sims = []
    for li in left_ids:
        i = node_index[li]
        for rj in right_ids:
            j = node_index[rj]
            sims.append(kernel[i, j])
    return 1.0 - (sum(sims) / len(sims))


# ----------------------------------------------------------------------
# Hybrid split decision
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑Gini‑RBF split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def hybrid_split_decision(parent_counts: Counter,
                          left_counts: Counter,
                          right_counts: Counter,
                          left_ids: List[Hashable],
                          right_ids: List[Hashable],
                          features: Dict[Hashable, Sequence[float]],
                          delta: float,
                          n_samples: int,
                          alpha: float = 0.6,
                          beta: float = 0.4,
                          rbf_epsilon: float = 1.0,
                          tie_threshold: float = 0.05) -> SplitDecision:
    """
    Combine Gini gain with RBF‑based dissimilarity, then apply Hoeffding bound.

    Parameters
    ----------
    parent_counts, left_counts, right_counts : Counter
        Class frequency counters.
    left_ids, right_ids : List[Hashable]
        Instance identifiers belonging to each child node.
    features : dict
        Mapping from instance id to its feature vector.
    delta : float
        Desired error probability for the Hoeffding test.
    n_samples : int
        Number of observations seen at the current node (used for ε).
    alpha, beta : float
        Weighting coefficients (α+β should be 1.0).
    rbf_epsilon : float
        Bandwidth parameter of the Gaussian kernel.
    tie_threshold : float
        Minimum ε below which we split even if gains are close.

    Returns
    -------
    SplitDecision
        Whether to split and diagnostic information.
    """
    if not (0 <= alpha <= 1 and 0 <= beta <= 1 and math.isclose(alpha + beta, 1.0)):
        raise ValueError("alpha and beta must be non‑negative and sum to 1")

    # 1️⃣ Gini component
    gini_gain_val = gini_gain(parent_counts, left_counts, right_counts)

    # 2️⃣ RBF component – compute kernel once for efficiency
    K, nodes = rbf_kernel_matrix(features, epsilon=rbf_epsilon)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    rbf_dissim = average_rbf_dissimilarity(left_ids, right_ids, K, node_index)

    # 3️⃣ Hybrid gain (both terms lie in [0,1])
    hybrid_gain = alpha * gini_gain_val + beta * rbf_dissim

    # 4️⃣ Hoeffding test – the hybrid gain is bounded by 1, so range_=1
    eps = hoeffding_bound(range_=1.0, delta=delta, n=n_samples)

    # In a real tree we would compare best vs second‑best gain.
    # Here we treat the alternative gain as 0.
    gain_gap = hybrid_gain - 0.0
    should = gain_gap > eps or eps < tie_threshold

    reason = ("split" if should else "no split") + f"; gain={hybrid_gain:.4f}, ε={eps:.4f}"
    return SplitDecision(should_split=should,
                         epsilon=eps,
                         gain_gap=gain_gap,
                         reason=reason)


# ----------------------------------------------------------------------
# Example helper – builds a toy node and runs the hybrid test
# ----------------------------------------------------------------------
def _toy_example() -> None:
    """Run a minimal smoke test on synthetic data."""
    random.seed(42)
    np.random.seed(42)

    # 10 samples, binary class, 2‑D features
    n = 10
    ids = list(range(n))
    features = {i: np.random.rand(2).tolist() for i in ids}
    labels = {i: random.choice([0, 1]) for i in ids}

    # Simple threshold on first feature to create a split
    threshold = 0.5
    left_ids = [i for i in ids if features[i][0] <= threshold]
    right_ids = [i for i in ids if features[i][0] > threshold]

    # Counters
    parent_counts = Counter(labels.values())
    left_counts = Counter(labels[i] for i in left_ids)
    right_counts = Counter(labels[i] for i in right_ids)

    decision = hybrid_split_decision(
        parent_counts=parent_counts,
        left_counts=left_counts,
        right_counts=right_counts,
        left_ids=left_ids,
        right_ids=right_ids,
        features=features,
        delta=0.05,
        n_samples=n,
        alpha=0.6,
        beta=0.4,
        rbf_epsilon=1.0,
        tie_threshold=0.02,
    )
    print("Hybrid split decision:", decision)


if __name__ == "__main__":
    _toy_example()