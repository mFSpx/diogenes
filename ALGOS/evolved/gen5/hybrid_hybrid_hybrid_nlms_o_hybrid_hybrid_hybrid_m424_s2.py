# DARWIN HAMMER — match 424, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s2.py (gen4)
# born: 2026-05-29T23:28:55Z

"""Hybrid NLMS‑LSM Minimum‑Cost Tree
===================================

Parents
-------
* **hybrid_nlms_omni_chaotic_sprint_m59_s3.py** – Normalized Least‑Mean‑Squares (NLMS) predictor.
* **hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py** – Latent‑Semantic‑Model (LSM) text vectors and a
  minimum‑cost spanning‑tree construction with Bayesian‑inspired edge weighting.

Mathematical Bridge
-------------------
For an unordered pair of nodes *e = (i, j)* we define a hybrid edge weight


d_ij      = Euclidean distance between the geometric coordinates of i and j
c_ij      = epistemic‑certainty factor supplied by the user (0 ≤ c ≤ 1)
p_i, p_j  = NLMS decision scores (scalar) associated with the two nodes
v_i, v_j  = LSM vectors derived from free‑form textual descriptors of the nodes
ℓ_ij      = cosine similarity(v_i, v_j)  ∈ [‑1, 1]   (used as a likelihood)
π_ij      = (p_i + p_j) / (p_i + p_j + ε)          (Bayesian prior)
ϕ_ij      = c_ij * 0.1                            (false‑positive term)

m_ij      = bayes_marginal(π_ij, 1‑ℓ_ij, ϕ_ij)    (marginal probability of error)
w_ij      = d_ij * (1 - m_ij) + ε                (final hybrid weight)


The NLMS scores provide a data‑driven prior, the LSM cosine similarity supplies a
likelihood, and the epistemic‑certainty factor contributes a small false‑positive
bias.  The resulting scalar `w_ij` is then fed to a classic Prim‑MST algorithm,
yielding a minimum‑cost spanning tree that respects both geometric, linguistic,
and adaptive‑filter information.

The module below implements the three core ingredients:

1. **NLMS prediction / update** – `nlms_predict` and `nlms_update`.
2. **LSM vector extraction** – `lsm_vector`.
3. **Hybrid edge weighting & MST construction** – `hybrid_edge_weight`,
   `bayes_marginal`, and `hybrid_minimum_spanning_tree`.

All functions are pure NumPy / std‑lib and can be used independently.  A tiny
smoke‑test is provided at the bottom of the file."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 1. NLMS core (parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the linear prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single Normalized Least‑Mean‑Squares (NLMS) adaptation step.

    Parameters
    ----------
    weights : np.ndarray
        Current filter coefficient vector (shape (n,)).
    x : np.ndarray
        Input regressor vector (shape (n,)).
    target : float
        Desired scalar response.
    mu : float, optional
        Step‑size (learning rate).  Typical range (0, 2).
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated coefficient vector.
    error : float
        Instantaneous prediction error (target – prediction).
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    norm_factor = (x @ x) + eps
    adaptation = (mu / norm_factor) * error * x
    new_weights = weights + adaptation
    return new_weights, error


# ----------------------------------------------------------------------
# 2. LSM utilities (parent B)
# ----------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """Very small word tokenizer – keeps only lower‑case alphabetic tokens."""
    return [tok for tok in text.lower().split() if tok.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Produce a simple term‑frequency (TF) representation of *text*.

    The output is a dictionary mapping token → relative frequency.
    """
    tokens = _tokenize(text)
    total = max(1, len(tokens))
    freq: Dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0.0) + 1.0 / total
    return freq


def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF vectors."""
    # Intersection contributes to the dot product
    dot = sum(v1[t] * v2[t] for t in v1 if t in v2)
    norm1 = math.sqrt(sum(val * val for val in v1.values()))
    norm2 = math.sqrt(sum(val * val for val in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


# ----------------------------------------------------------------------
# 3. Bayesian marginal helper (shared)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, fp: float, eps: float = 1e-9) -> float:
    """
    Compute the Bayesian marginal probability of error.

    posterior ∝ prior * likelihood
    denominator = prior*likelihood + (1‑prior)*fp
    """
    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * fp + eps
    return numerator / denominator


# ----------------------------------------------------------------------
# 4. Hybrid edge weighting
# ----------------------------------------------------------------------
def hybrid_edge_weight(
    i: int,
    j: int,
    positions: List[Tuple[float, float]],
    nlms_scores: List[float],
    lsm_vectors: List[Dict[str, float]],
    cert_factors: List[float],
    eps: float = 1e-9,
) -> float:
    """
    Compute the hybrid weight for edge (i, j) according to the bridge formula.

    Parameters
    ----------
    i, j : int
        Node indices.
    positions : list of (x, y)
        Geometric coordinates.
    nlms_scores : list of float
        Decision scores produced by an NLMS filter.
    lsm_vectors : list of dict
        Sparse LSM vectors for each node.
    cert_factors : list of float
        Epistemic‑certainty factor per node (0 ≤ c ≤ 1).
    eps : float
        Small constant to keep the weight strictly positive.

    Returns
    -------
    weight : float
        Hybrid edge weight.
    """
    # Euclidean distance
    xi, yi = positions[i]
    xj, yj = positions[j]
    d_ij = math.hypot(xi - xj, yi - yj)

    # Prior from NLMS scores
    pi = nlms_scores[i]
    pj = nlms_scores[j]
    prior = (pi + pj) / (pi + pj + eps)

    # Likelihood from LSM cosine similarity (shifted to [0,1])
    sim = cosine_similarity(lsm_vectors[i], lsm_vectors[j])  # ∈ [‑1,1]
    likelihood = 1.0 - ((sim + 1.0) / 2.0)  # map to error‑likelihood ∈ [0,1]

    # False‑positive term from epistemic certainty
    ci = cert_factors[i]
    cj = cert_factors[j]
    fp = ((ci + cj) / 2.0) * 0.1

    # Bayesian marginal
    m_ij = bayes_marginal(prior, likelihood, fp, eps)

    # Final hybrid weight
    weight = d_ij * (1.0 - m_ij) + eps
    return weight


# ----------------------------------------------------------------------
# 5. Minimum‑Cost Spanning Tree (Prim) using hybrid weights
# ----------------------------------------------------------------------
def hybrid_minimum_spanning_tree(
    positions: List[Tuple[float, float]],
    nlms_scores: List[float],
    texts: List[str],
    cert_factors: List[float],
) -> List[Tuple[int, int, float]]:
    """
    Build a minimum‑cost spanning tree where edge costs are given by
    `hybrid_edge_weight`.

    Returns a list of edges (u, v, weight) in the order they are added.
    """
    n = len(positions)
    if n == 0:
        return []

    # Pre‑compute LSM vectors once
    lsm_vectors = [lsm_vector(t) for t in texts]

    # Prim's algorithm
    in_tree = [False] * n
    min_edge: List[float] = [math.inf] * n
    parent: List[int] = [-1] * n

    # Start from node 0
    min_edge[0] = 0.0

    mst_edges: List[Tuple[int, int, float]] = []

    for _ in range(n):
        # Select the vertex with the smallest connecting edge
        u = -1
        best = math.inf
        for v in range(n):
            if not in_tree[v] and min_edge[v] < best:
                best = min_edge[v]
                u = v
        if u == -1:
            break  # disconnected graph (should not happen)

        in_tree[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u, min_edge[u]))

        # Update neighbours
        for v in range(n):
            if in_tree[v]:
                continue
            w = hybrid_edge_weight(
                u,
                v,
                positions,
                nlms_scores,
                lsm_vectors,
                cert_factors,
            )
            if w < min_edge[v]:
                min_edge[v] = w
                parent[v] = u

    return mst_edges


# ----------------------------------------------------------------------
# 6. Small smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create a tiny synthetic dataset
    num_nodes = 6
    positions = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_nodes)]

    # NLMS scores – pretend they are outputs of an adaptive filter
    dim = 3
    nlms_weights = np.random.randn(dim)
    nlms_scores = []
    for _ in range(num_nodes):
        x = np.random.randn(dim)
        nlms_scores.append(nlms_predict(nlms_weights, x))

    # Textual descriptors (simple English phrases)
    sample_texts = [
        "bright sunny day",
        "rainy cloudy evening",
        "mountainous terrain with rocks",
        "calm lake surface",
        "dense forest with tall trees",
        "urban city skyline at night",
    ]

    # Epistemic‑certainty factors (random but bounded)
    cert_factors = [random.random() for _ in range(num_nodes)]

    # Build the hybrid MST
    mst = hybrid_minimum_spanning_tree(
        positions=positions,
        nlms_scores=nlms_scores,
        texts=sample_texts,
        cert_factors=cert_factors,
    )

    # Output result
    print("Hybrid Minimum‑Cost Spanning Tree (node_a, node_b, weight):")
    for u, v, w in mst:
        print(f"({u}, {v}) -> {w:.4f}")

    # Demonstrate a single NLMS adaptation step
    x_demo = np.random.randn(dim)
    target_demo = 0.5
    new_weights, err = nlms_update(nlms_weights, x_demo, target_demo)
    print("\nNLMS update demo:")
    print(f"  error = {err:.6f}")
    print(f"  weight norm change = {np.linalg.norm(new_weights - nlms_weights):.6f}")