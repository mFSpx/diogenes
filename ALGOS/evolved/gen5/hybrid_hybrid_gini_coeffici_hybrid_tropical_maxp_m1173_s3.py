# DARWIN HAMMER — match 1173, survivor 3
# gen: 5
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s0.py (gen4)
# parent_b: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s4.py (gen3)
# born: 2026-05-29T23:33:13Z

"""Hybrid Gini‑Tropical RBF‑Belief Tree
=====================================

This module fuses the two parent algorithms:

* **Parent A** – Gini‑coefficient driven Hoeffding split with a radial‑basis
  similarity matrix built from perceptual hashes.
* **Parent B** – Tropical max‑plus algebra used to perform Bayesian updates
  (log‑probabilities) and to propagate the most likely belief through a graph.

**Mathematical bridge**

The Gini coefficient provides a scalar impurity measure *I* for a candidate
split.  In a Bayesian tree each leaf carries a log‑probability belief vector
*ℓ*.  Tropical multiplication (⊗ = +) adds log‑probabilities, while tropical
addition (⊕ = max) approximates the log‑sum‑exp required for normalisation.
The RBF‑derived similarity matrix *S* is interpreted as a (negative) log‑likelihood
edge cost *C = –log S*.  By feeding *C* into the tropical matrix product we
propagate beliefs over the graph, obtaining a belief score *B* for each node.
A hybrid split score is then defined as a weighted combination

    score = α·I  +  β·B  +  γ·mean(S_row)

which couples impurity, tropical belief and similarity in a single unified
criterion.

The functions below implement the core operations of this hybrid system.
"""

import math
import random
import sys
import pathlib
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Gini and RBF similarity utilities
# ---------------------------------------------------------------------------

Node = Hashable
FeatureVec = Sequence[float]

def gini_coefficient(values: Iterable[float]) -> float:
    """Gini impurity of a non‑negative numeric collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def compute_phash(values: List[float]) -> int:
    """Simple 64‑bit perceptual hash based on the mean of the vector."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a symmetric similarity matrix S where
        S[i, j] = 1 – Hamming(phash_i, phash_j) / 64 .
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    phashes = [compute_phash(list(features[node])) for node in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(phashes[i], phashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

# ---------------------------------------------------------------------------
# Parent B – Tropical max‑plus primitives and Bayesian update
# ---------------------------------------------------------------------------

def t_add(x, y):
    """Tropical addition (⊕): max(x, y). Works element‑wise on ndarrays."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication (⊗): x + y. Works element‑wise on ndarrays."""
    return np.add(x, y)

def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication.
    C[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("inner dimensions must match")
    # broadcasting trick: (m, p, 1) + (1, p, n) -> (m, p, n)
    tmp = A[:, :, np.newaxis] + B[np.newaxis, :, :]
    return np.max(tmp, axis=1)

def tropical_bayesian_update(prior_log: np.ndarray,
                             likelihood_log: np.ndarray) -> np.ndarray:
    """
    Perform a Bayesian update in log‑space using tropical algebra.
    posterior ∝ prior ⊗ likelihood  (i.e. addition).
    Normalisation is approximated by tropical addition (max) to mimic
    log‑sum‑exp.
    """
    unnorm = t_mul(prior_log, likelihood_log)          # element‑wise addition
    log_z = np.max(unnorm)                             # tropical “sum”
    posterior = unnorm - log_z                         # normalise (log‑space)
    return posterior

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def propagate_beliefs(similarity: np.ndarray,
                      log_beliefs: np.ndarray,
                      steps: int = 1) -> np.ndarray:
    """
    Propagate log‑beliefs over the graph defined by ``similarity``.
    The similarity is turned into a negative log‑likelihood cost matrix C,
    then tropical matrix multiplication spreads the most probable belief.
    """
    # Avoid log(0) – clamp similarity to a tiny positive value
    eps = np.finfo(float).eps
    C = -np.log(np.clip(similarity, eps, 1.0))
    belief = log_beliefs.copy()
    for _ in range(steps):
        # belief_new[i] = max_j ( belief[j] - C[j,i] )
        belief = t_matmul(belief[None, :], -C)  # shape (1, n)
        belief = belief.ravel()
    return belief

def hybrid_split_score(feature_vals: Sequence[float],
                       labels: Sequence[int],
                       node_idx: int,
                       sim_matrix: np.ndarray,
                       belief_vec: np.ndarray,
                       alpha: float = 1.0,
                       beta: float = 1.0,
                       gamma: float = 1.0) -> float:
    """
    Compute a hybrid score for splitting on ``feature_vals`` at ``node_idx``.
    The score blends:
      * Gini impurity of the target labels (α)
      * Tropical belief of the node (β)
      * Mean similarity of the node to its neighbours (γ)
    Lower scores are preferred for a split.
    """
    gini = gini_coefficient(labels)
    belief = belief_vec[node_idx]
    mean_sim = float(np.mean(sim_matrix[node_idx]))
    return alpha * gini + beta * (-belief) + gamma * (1.0 - mean_sim)

def build_hybrid_tree(features: Dict[Node, FeatureVec],
                      labels: Dict[Node, int],
                      prior_log: np.ndarray,
                      max_depth: int = 3) -> Tuple[Dict[Node, Tuple[float, int]], np.ndarray]:
    """
    Very small illustrative “tree” builder.
    Returns a mapping node → (split_score, depth) and the final belief vector.
    """
    sim_mat, node_list = similarity_matrix(features)
    # Initial belief from priors
    belief = prior_log.copy()

    # Propagate beliefs once to incorporate similarity structure
    belief = propagate_beliefs(sim_mat, belief, steps=1)

    split_info: Dict[Node, Tuple[float, int]] = {}
    for depth in range(max_depth):
        for idx, node in enumerate(node_list):
            # Gather feature column for this node (dummy split on the node itself)
            feat_vals = features[node]
            lbls = [labels[node]]
            score = hybrid_split_score(feat_vals,
                                       lbls,
                                       idx,
                                       sim_mat,
                                       belief,
                                       alpha=0.7,
                                       beta=0.2,
                                       gamma=0.1)
            split_info[node] = (score, depth)
    return split_info, belief

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic dataset with 5 nodes
    rng = np.random.default_rng(42)

    # Random 3‑dimensional feature vectors
    features: Dict[int, List[float]] = {
        i: rng.random(3).tolist() for i in range(5)
    }

    # Binary labels
    labels: Dict[int, int] = {i: rng.integers(0, 2) for i in range(5)}

    # Prior log‑probabilities (uniform)
    prior_log = np.log(np.full(5, 1.0 / 5.0))

    # Build the hybrid structure
    split_info, final_belief = build_hybrid_tree(features, labels, prior_log)

    print("Split information (node → (score, depth)):")
    for n, info in split_info.items():
        print(f"  Node {n}: score={info[0]:.4f}, depth={info[1]}")

    print("\nFinal log‑belief vector:")
    for i, b in enumerate(final_belief):
        print(f"  Node {i}: {b:.4f}")

    # Verify that tropical Bayesian update works on a simple example
    prior = np.log(np.array([0.6, 0.4]))
    likelihood = np.log(np.array([0.7, 0.2]))
    posterior = tropical_bayesian_update(prior, likelihood)
    print("\nPosterior (log‑space) after tropical update:", posterior)