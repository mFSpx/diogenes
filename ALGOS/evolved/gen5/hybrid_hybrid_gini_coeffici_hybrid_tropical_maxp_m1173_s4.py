# DARWIN HAMMER — match 1173, survivor 4
# gen: 5
# parent_a: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s0.py (gen4)
# parent_b: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s4.py (gen3)
# born: 2026-05-29T23:33:13Z

"""Hybrid Gini‑Tropical RBF Tree
================================

This module fuses two evolutionary parents:

* **Parent A** – ``gini_coefficient.py`` + RBF‑based similarity used for
  Hoeffding‑tree splitting.
* **Parent B** – ``tropical_maxplus.py`` + Bayesian/entropy decision scoring.

**Mathematical bridge**

The Gini coefficient measures class‑distribution inequality at a node and
produces a scalar that can be added (in log‑space) to a belief score.
Probabilistic beliefs are stored as log‑probabilities; tropical multiplication
(``⊗ = +``) corresponds to ordinary addition of log‑probabilities, while tropical
addition (``⊕ = max``) approximates the log‑sum‑exp needed for Bayesian
marginalisation.  By converting the RBF similarity matrix into a tropical weight
matrix (negative log‑similarity) we can propagate log‑beliefs through the graph
with a single tropical matrix multiplication.  The final hybrid split score
adds the Gini term to the propagated belief, yielding a decision metric that
captures both class‑distribution inequality and most‑probable belief propagation.

The three core functions below demonstrate this fusion:
``hybrid_similarity_matrix`` – RBF similarity → tropical weights,
``tropical_propagate`` – tropical matrix multiplication of log‑beliefs,
``hybrid_split_score`` – Gini‑augmented decision score.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Gini & RBF utilities
# ---------------------------------------------------------------------------

Node = Hashable
FeatureVec = Sequence[float]

def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative value list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))

def hybrid_similarity_matrix(
    features: Dict[Node, FeatureVec],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute an RBF similarity matrix S where
        S[i, j] = exp(- (epsilon * ||f_i - f_j||)^2 ).

    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        fi = features[ni]
        for j, nj in enumerate(nodes):
            fj = features[nj]
            d = euclidean(fi, fj)
            S[i, j] = gaussian(d, epsilon=epsilon)
    return S, nodes

# ---------------------------------------------------------------------------
# Parent B – Tropical (max‑plus) primitives
# ---------------------------------------------------------------------------

def t_add(x, y):
    """Tropical addition (⊕) – element‑wise max."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication (⊗) – element‑wise addition."""
    return np.add(x, y)

def t_matmul(A, B):
    """
    Tropical matrix multiplication.

    (A ⊗ B)[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Broadcast addition over the k‑dimension and take max.
    return np.max(A[:, :, None] + B[None, :, :], axis=1)

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def tropical_propagate(
    features: Dict[Node, FeatureVec],
    adjacency: Dict[Node, Set[Node]],
    log_prior: np.ndarray,
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Propagate log‑beliefs through the graph using tropical algebra.

    Steps
    -----
    1. Build an RBF similarity matrix S from node features.
    2. Convert S to a tropical weight matrix W = -log(S + 1e-12)
       (high similarity ⇒ low tropical cost).
    3. Encode the admissible edges from ``adjacency`` as a mask M where
       allowed edges have cost 0 and disallowed edges have +inf.
    4. Combine similarity cost and edge mask: T = W ⊗ M  (element‑wise add).
    5. Perform one step of tropical matrix multiplication:
           belief = log_prior ⊗ T
    Returns the belief matrix and the node ordering.
    """
    eps = 1e-12
    S, nodes = hybrid_similarity_matrix(features, epsilon=epsilon)
    W = -np.log(S + eps)                     # tropical costs
    n = len(nodes)

    # Edge mask M: 0 for allowed edge, +inf otherwise.
    M = np.full((n, n), np.inf, dtype=float)
    node_index = {node: i for i, node in enumerate(nodes)}
    for u, neighbours in adjacency.items():
        ui = node_index[u]
        for v in neighbours:
            vi = node_index[v]
            M[ui, vi] = 0.0

    T = t_mul(W, M)                         # tropical addition = ordinary add
    belief = t_matmul(log_prior, T)         # propagate
    return belief, nodes

def hybrid_split_score(
    node: Node,
    node_labels: List[int],
    belief_matrix: np.ndarray,
    node_order: List[Node],
) -> float:
    """
    Compute a hybrid decision score for ``node`` by adding the
    Gini coefficient of its label distribution to the tropical belief.

    The belief is taken from ``belief_matrix`` (assumed shape (1, n)).
    """
    idx = node_order.index(node)
    gini = gini_coefficient(node_labels)
    # Guard against log(0)
    eps = 1e-12
    belief = belief_matrix[0, idx]
    return belief + math.log(gini + eps)

def hybrid_tree_step(
    features: Dict[Node, FeatureVec],
    adjacency: Dict[Node, Set[Node]],
    labels_per_node: Dict[Node, List[int]],
    prior_probs: List[float],
) -> Dict[Node, float]:
    """
    One high‑level hybrid step:
      * Convert priors to log‑space.
      * Propagate beliefs tropically.
      * Produce a Gini‑augmented split score per node.

    Returns a mapping ``node -> score``.
    """
    eps = 1e-12
    log_prior = np.log(np.array(prior_probs, dtype=float) + eps).reshape(1, -1)
    belief, order = tropical_propagate(features, adjacency, log_prior)
    scores = {}
    for node in order:
        scores[node] = hybrid_split_score(
            node,
            labels_per_node.get(node, []),
            belief,
            order,
        )
    return scores

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Small synthetic example with three nodes.
    random.seed(42)
    np.random.seed(42)

    # Feature vectors (2‑D)
    feats = {
        "A": np.random.rand(2).tolist(),
        "B": np.random.rand(2).tolist(),
        "C": np.random.rand(2).tolist(),
    }

    # Undirected adjacency
    adj = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B"},
    }

    # Labels per node (binary classification example)
    lbls = {
        "A": [0, 0, 1],
        "B": [1, 1, 1, 0],
        "C": [0, 0],
    }

    # Uniform prior probabilities over the three nodes
    priors = [1 / 3, 1 / 3, 1 / 3]

    scores = hybrid_tree_step(feats, adj, lbls, priors)
    for node, sc in scores.items():
        print(f"Node {node}: hybrid split score = {sc:.4f}")