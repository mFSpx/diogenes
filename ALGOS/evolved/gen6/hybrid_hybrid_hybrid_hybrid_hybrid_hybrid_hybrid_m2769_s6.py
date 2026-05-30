# DARWIN HAMMER — match 2769, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1978_s2.py (gen4)
# born: 2026-05-29T23:45:43Z

"""Hybrid NLMS‑RBF‑Bayesian Router
================================

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – an NLMS adaptive filter whose weight vector is embedded in a
  radial‑basis‑function (RBF) similarity graph and later processed by a
  minimum‑cost‑tree (MCT) routine.
* **Parent B** – a Bayesian router that updates prior probabilities of
  geometric “multivector” objects using an RBF‑based similarity matrix.

The **mathematical bridge** is the Gaussian RBF similarity that appears in both
parents together with the binary perceptual hash (`phash`) utilities.  In the
hybrid we

1. treat each filter weight as a 2‑D point ``(i, w_i)`` so that the Euclidean
   distance used by the RBF is identical to the one in the Bayesian similarity
   matrix,
2. use the resulting similarity matrix as a *likelihood* in a Bayesian update of
   per‑weight prior probabilities,
3. modulate the NLMS weight update with those posterior probabilities and feed
   the resulting weighted graph into the original minimum‑cost‑tree algorithm.

The three public functions below illustrate the combined workflow:
`nlms_rbf_update`, `bayesian_prior_update`, and `hybrid_minimum_cost_tree`.  """

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared low‑level primitives (present in both parents)
# ----------------------------------------------------------------------


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash – 64‑bit at most."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# NLMS core (Parent A)
# ----------------------------------------------------------------------


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = wᵀx."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised LMS weight update.

    Returns the new weight vector and the instantaneous error.
    """
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + mu * error * x / power
    return next_weights, error


# ----------------------------------------------------------------------
# Bayesian core (Parent B)
# ----------------------------------------------------------------------


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Simple Bayesian posterior ∝ prior × likelihood.
    The result is normalised to sum to 1.
    """
    unnorm = prior * likelihood
    total = float(unnorm.sum())
    if total == 0.0:
        # avoid division by zero – revert to uniform prior
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """Graph node holding weight and its posterior probability."""
    id: int
    weight: float
    prob: float  # posterior probability (Bayesian)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def nlms_rbf_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform an NLMS update and then smooth the resulting weights with an
    RBF‑based similarity filter (the “RBF bridge”).

    The smoothing step replaces each weight w_i by a weighted average of its
    neighbours, where the weights are Gaussian similarities of the scalar
    values.  This mirrors the similarity‑graph construction of Parent A.
    """
    # 1️⃣ NLMS step
    updated, error = nlms_update(weights, x, target, mu, eps)

    # 2️⃣ RBF smoothing (pairwise Gaussian similarity on scalar weights)
    n = updated.size
    smoothed = np.empty_like(updated)
    for i in range(n):
        sims = np.array(
            [
                gaussian(abs(updated[i] - updated[j]))
                for j in range(n)
                if j != i
            ]
        )
        neigh_weights = np.delete(updated, i)
        # Normalised weighted average
        if sims.sum() == 0:
            smoothed[i] = updated[i]
        else:
            smoothed[i] = float(np.dot(sims, neigh_weights) / sims.sum())
    return smoothed, error


def bayesian_prior_update(
    weights: np.ndarray,
    prior: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Build an RBF similarity matrix from the *geometric* embedding of the
    weights (point i → (i, w_i)).  Use the row‑wise sum as a likelihood for each
    weight and perform a Bayesian update of the prior probabilities.
    """
    n = weights.size
    # Embed each weight as a 2‑D point (index, value)
    points = [(float(i), float(w)) for i, w in enumerate(weights)]

    # Compute similarity matrix S_ij = exp(-ε²‖p_i-p_j‖²)
    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean([points[i][0], points[i][1]], [points[j][0], points[j][1]])
            val = gaussian(dist, epsilon)
            S[i, j] = val
            S[j, i] = val

    # Likelihood for each weight = sum of similarities to all others (excluding self)
    likelihood = S.sum(axis=1) - np.diag(S)
    # Avoid zero likelihoods
    likelihood = np.where(likelihood == 0, 1e-12, likelihood)

    posterior = bayesian_update(prior, likelihood)
    return posterior


def construct_hybrid_graph(weights: np.ndarray, posterior: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a directed graph where edge weight = Gaussian similarity × posterior of
    the *target* node.  This merges Parent A's graph construction with the
    Bayesian probabilities from Parent B.
    """
    n = weights.size
    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        graph[i] = []
        for j in range(n):
            if i == j:
                continue
            sim = gaussian(abs(weights[i] - weights[j]))
            edge_weight = sim * posterior[j]  # modulate by posterior of destination
            graph[i].append((j, edge_weight))
    return graph


def hybrid_minimum_cost_tree(
    graph: Dict[int, List[Tuple[int, float]]]
) -> List[Tuple[int, int, float]]:
    """
    Prim's algorithm on the hybrid graph.  Returns a list of edges (src, dst,
    weight) forming the minimum‑cost spanning tree.
    """
    if not graph:
        return []

    start = next(iter(graph))
    visited = {start}
    edges: List[Tuple[int, int, float]] = []
    import heapq

    # priority queue of (weight, src, dst)
    frontier: List[Tuple[float, int, int]] = []
    for dst, w in graph[start]:
        heapq.heappush(frontier, (w, start, dst))

    while frontier and len(visited) < len(graph):
        w, src, dst = heapq.heappop(frontier)
        if dst in visited:
            continue
        visited.add(dst)
        edges.append((src, dst, w))
        for nxt, w2 in graph[dst]:
            if nxt not in visited:
                heapq.heappush(frontier, (w2, dst, nxt))

    return edges


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Synthetic data: 8‑dimensional NLMS filter
    dim = 8
    w = np.random.randn(dim) * 0.5
    prior = np.full(dim, 1.0 / dim)  # uniform prior

    # Input vector and target (simple linear relation with noise)
    x = np.random.randn(dim)
    true_w = np.random.randn(dim)
    target = float(np.dot(true_w, x) + np.random.randn() * 0.1)

    # Hybrid iteration
    for step in range(5):
        w, err = nlms_rbf_update(w, x, target, mu=0.3)
        prior = bayesian_prior_update(w, prior, epsilon=0.8)
        graph = construct_hybrid_graph(w, prior)
        tree = hybrid_minimum_cost_tree(graph)

        print(f"Step {step+1:02d} | error={err:.4f} | weight hash={compute_phash(w.tolist())}")
        print(f"  Posterior probs: {np.round(prior, 3)}")
        print(f"  Tree edges ({len(tree)}): {tree}\n")
    sys.exit(0)