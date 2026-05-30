# DARWIN HAMMER — match 3109, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1481_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m2318_s2.py (gen6)
# born: 2026-05-29T23:47:51Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1481_s1.py (geometric product + certainty weighting + lazy random walk)
- hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m2318_s2.py (tropical distance + Hoeffding bound + Gini coefficient)

Mathematical Bridge:
Both parents operate on a set of feature vectors.  The first parent embeds vectors into a
Clifford geometric product and scales them by a certainty weight.  The second parent
measures pair‑wise similarity with a distance metric (tropical distance) and uses
statistical bounds (Hoeffding) modulated by a Gini‑based impurity measure to decide
whether to split a node.  The fusion therefore:
1. Computes a certainty‑weighted geometric product matrix (scalar part = dot product).
2. Interprets the resulting similarity matrix as a weighted graph.
3. Applies a lazy random‑walk to obtain a probability distribution over nodes.
4. Uses tropical distance on the walk‑derived node embeddings and a Gini‑modulated
   Hoeffding bound to produce a split decision.

The three core functions below demonstrate this pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0..10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

# ----------------------------------------------------------------------
# 1. Geometric product (scalar part only) with certainty weighting
# ----------------------------------------------------------------------
def certainty_weighted_geometric_product(
    a: np.ndarray, b: np.ndarray, flag: CertaintyFlag
) -> float:
    """
    Compute the scalar part (dot product) of the geometric product between
    two vectors and weight it by the certainty factor w = confidence_bps / 10000.
    """
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    w = flag.confidence_bps / 10000.0
    return w * float(np.dot(a, b))

def similarity_matrix(
    features: np.ndarray, flag: CertaintyFlag
) -> np.ndarray:
    """
    Build an N×N similarity matrix S where S[i,j] is the certainty‑weighted
    geometric product between feature i and feature j.
    """
    n = features.shape[0]
    S = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = certainty_weighted_geometric_product(
                features[i], features[j], flag
            )
            S[i, j] = val
            S[j, i] = val
    return S

# ----------------------------------------------------------------------
# 2. Lazy random walk on the similarity graph
# ----------------------------------------------------------------------
def lazy_random_walk(
    adjacency: np.ndarray, steps: int = 10, laziness: float = 0.15
) -> np.ndarray:
    """
    Perform a lazy random walk on a weighted adjacency matrix.
    The transition matrix is:
        T = (1 - laziness) * D^{-1} A + laziness * I
    where D is the degree matrix.
    Returns the stationary distribution after ``steps`` multiplications
    of an initial uniform distribution.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    n = adjacency.shape[0]
    # Ensure non‑negative weights
    A = np.maximum(adjacency, 0.0)
    degree = A.sum(axis=1, keepdims=True)
    # Avoid division by zero
    degree[degree == 0] = 1.0
    P = A / degree  # row‑stochastic
    T = (1.0 - laziness) * P + laziness * np.eye(n)
    # start from uniform distribution
    pi = np.full(n, 1.0 / n)
    for _ in range(steps):
        pi = pi @ T
    return pi

# ----------------------------------------------------------------------
# 3. Hybrid split decision (tropical distance + Gini‑modulated Hoeffding bound)
# ----------------------------------------------------------------------
def tropical_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Maximum absolute coordinate difference (L∞ norm)."""
    return float(np.max(np.abs(x - y)))

def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini impurity for a non‑negative vector."""
    xs = np.array([float(v) for v in values])
    if xs.size == 0 or xs.sum() == 0:
        return 0.0
    if np.any(xs < 0):
        raise ValueError("values must be non‑negative")
    n = xs.size
    sorted_x = np.sort(xs)
    cum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("invalid parameters for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def modulated_hoeffding_bound(r: float, delta: float, n: int, gini: float) -> float:
    """
    Modulate the Hoeffding bound by an exponential factor that depends on Gini.
    The factor shrinks the bound when impurity is high (large Gini).
    """
    epsilon = 0.1
    gamma = math.exp(-(epsilon * gini) ** 2)
    return hoeffding_bound(r, delta, n) * gamma

def hybrid_split_decision(
    features: np.ndarray,
    flag: CertaintyFlag,
    delta: float = 0.05,
    n_observations: int = 100,
) -> Tuple[bool, float, float]:
    """
    Combine the three pillars:
    * Build a certainty‑weighted similarity matrix (geometric product).
    * Derive node embeddings via a lazy random walk.
    * Compute tropical distances between the resulting probability vector and each node.
      Apply a Gini‑modulated Hoeffding bound to decide whether a split is warranted.

    Returns (should_split, epsilon, gain_gap).
    """
    # 1. Similarity matrix
    S = similarity_matrix(features, flag)

    # 2. Random‑walk distribution (probability per node)
    pi = lazy_random_walk(S, steps=20, laziness=0.2)

    # 3. Pairwise tropical distances between the walk distribution and each
    #    node's similarity row (treated as a feature vector)
    distances = np.array(
        [tropical_distance(pi, S[i]) for i in range(S.shape[0])]
    )
    max_dist = float(distances.max())
    min_dist = float(distances.min())
    gain_gap = max_dist - min_dist

    # 4. Gini coefficient on the certainty‑weighted similarity rows
    gini_vals = np.array([gini_coefficient(row) for row in S])

    # 5. Modulated Hoeffding bound per node, then aggregate
    mod_bounds = np.array(
        [
            modulated_hoeffding_bound(
                r=max_dist, delta=delta, n=n_observations, gini=gini_vals[i]
            )
            for i in range(S.shape[0])
        ]
    )
    epsilon = float(mod_bounds.max())
    should_split = epsilon > 0.0 and gain_gap > epsilon

    return should_split, epsilon, gain_gap

# ----------------------------------------------------------------------
# Additional demonstration function
# ----------------------------------------------------------------------
def hybrid_process(
    features: np.ndarray,
    flag: CertaintyFlag,
    delta: float = 0.05,
    n_observations: int = 100,
) -> dict:
    """
    Run the full hybrid pipeline and return a dictionary with intermediate
    results for inspection.
    """
    S = similarity_matrix(features, flag)
    pi = lazy_random_walk(S, steps=30, laziness=0.1)
    should_split, epsilon, gain_gap = hybrid_split_decision(
        features, flag, delta, n_observations
    )
    return {
        "similarity_matrix": S,
        "walk_distribution": pi,
        "should_split": should_split,
        "epsilon": epsilon,
        "gain_gap": gain_gap,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic feature set (10 nodes, 5‑dimensional)
    np.random.seed(42)
    num_nodes = 10
    dim = 5
    features = np.random.randn(num_nodes, dim)

    # Create a certainty flag (medium confidence)
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=6500,
        authority_class="synthetic",
        rationale="unit test",
    )

    result = hybrid_process(features, flag, delta=0.05, n_observations=200)

    # Simple sanity checks – they should all run without raising.
    assert isinstance(result["similarity_matrix"], np.ndarray)
    assert result["similarity_matrix"].shape == (num_nodes, num_nodes)
    assert isinstance(result["walk_distribution"], np.ndarray)
    assert np.isclose(result["walk_distribution"].sum(), 1.0, atol=1e-6)
    print("Hybrid process completed successfully.")
    print(f"Should split? {result['should_split']}")
    print(f"Epsilon: {result['epsilon']:.6f}, Gain gap: {result['gain_gap']:.6f}")