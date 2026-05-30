# DARWIN HAMMER — match 2769, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_percep_m598_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1978_s2.py (gen4)
# born: 2026-05-29T23:45:43Z

"""Hybrid NLMS‑Bayesian RBF Fusion
=================================

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** provides an adaptive Normalised Least‑Mean‑Squares (NLMS) weight
  update together with a radial‑basis‑function (RBF) similarity graph.
* **Parent B** supplies a Bayesian update of prior probabilities for a set of
  geometric objects (here called *Multivectors*) using an RBF‑based similarity
  matrix.

The **mathematical bridge** is the Gaussian RBF kernel, which appears in both
parents for measuring similarity.  In the hybrid algorithm the RBF kernel is
used

1. to turn the current NLMS weight vector into a likelihood over the
   Multivector prototypes,
2. to weight the Bayesian posterior, and
3. to build a graph that can be traversed for downstream routing decisions.

The result is a single unified step that simultaneously adapts the filter
weights (NLMS) and updates the belief distribution over geometric objects
(Bayesian inference)."""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared elementary utilities (identical in both parents)
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
    """Perceptual hash – simple average‑threshold binarisation."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    """Graph node used in the RBF similarity graph (Parent A)."""
    id: int
    weight: float


class Multivector:
    """Light‑weight geometric object used in the Bayesian router (Parent B)."""
    def __init__(self, components: List[float]):
        self.components = np.asarray(components, dtype=np.float64)

    def distance(self, other: "Multivector") -> float:
        """Euclidean distance in component space."""
        return float(np.linalg.norm(self.components - other.components))


# ----------------------------------------------------------------------
# Core NLMS operations (Parent A)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised Least‑Mean‑Squares update.
    Returns the new weight vector and the instantaneous error.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + mu * error * x / power
    return new_weights, error


# ----------------------------------------------------------------------
# Bayesian RBF router (Parent B)
# ----------------------------------------------------------------------
def similarity_matrix(multivectors: List[Multivector]) -> np.ndarray:
    """
    Build an RBF similarity matrix S_ij = exp(-||v_i - v_j||²).
    The matrix is symmetric and has ones on the diagonal.
    """
    n = len(multivectors)
    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                val = 1.0
            else:
                d = multivectors[i].distance(multivectors[j])
                val = gaussian(d)
            S[i, j] = val
            S[j, i] = val
    return S


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Perform a Bayesian update: posterior ∝ prior * likelihood.
    Both inputs must be 1‑D arrays of the same length.
    The result is normalised to sum to 1.
    """
    if prior.shape != likelihood.shape:
        raise ValueError("prior and likelihood must share shape")
    unnorm = prior * likelihood
    total = float(unnorm.sum())
    if total == 0.0:
        # Avoid division by zero – fall back to uniform distribution
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total


# ----------------------------------------------------------------------
# Hybrid constructs
# ----------------------------------------------------------------------
def construct_rbf_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a fully connected graph where edge weights are Gaussian RBFs of the
    distance between scalar weight entries (Parent A).  The graph is useful for
    downstream minimum‑cost‑tree operations.
    """
    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(len(weights)):
        node = Node(i, float(weights[i]))
        graph[node.id] = []
        for j in range(len(weights)):
            if i == j:
                continue
            sim = gaussian(euclidean([weights[i]], [weights[j]]))
            graph[node.id].append((j, sim))
    return graph


def hybrid_nlms_bayes_step(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    prior: np.ndarray,
    prototypes: List[Multivector],
    mu: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One hybrid iteration:

    1. **NLMS** adapts the weight vector using the current sample (x, target).
    2. The updated weights are projected onto the prototype space:
       each prototype receives a likelihood proportional to an RBF of the
       distance between the weight vector and the prototype components.
    3. **Bayesian** posterior is obtained by combining the prior with the
       likelihood.

    Returns the new weight vector and the updated posterior distribution.
    """
    # ----- NLMS adaptation -----
    w_new, _ = nlms_update(weights, x, target, mu=mu)

    # ----- Build likelihood from RBF similarity -----
    # Treat the weight vector as a point in the same space as the prototypes.
    weight_mv = Multivector(w_new.tolist())
    likelihood = np.empty(len(prototypes), dtype=np.float64)
    for idx, proto in enumerate(prototypes):
        d = weight_mv.distance(proto)
        likelihood[idx] = gaussian(d)

    # Normalise likelihood to avoid degenerate scaling
    likelihood_sum = float(likelihood.sum())
    if likelihood_sum > 0:
        likelihood /= likelihood_sum
    else:
        likelihood = np.full_like(likelihood, 1.0 / len(likelihood))

    # ----- Bayesian posterior update -----
    posterior = bayesian_update(prior, likelihood)

    return w_new, posterior


def hybrid_predict(
    weights: np.ndarray,
    x: np.ndarray,
    prior: np.ndarray,
    prototypes: List[Multivector],
) -> float:
    """
    Combine a linear prediction with a Bayesian expectation over prototypes.
    The final output is a convex combination:

        y = (1 - α) * (w·x) + α * Σ_i posterior_i * f_i(x)

    where f_i(x) is a simple dot product between prototype i and the input.
    The mixing coefficient α is derived from the entropy of the posterior
    (high uncertainty → larger α).
    """
    # Linear part
    lin = predict(weights, x)

    # Posterior (reuse similarity as likelihood)
    weight_mv = Multivector(weights.tolist())
    likelihood = np.empty(len(prototypes), dtype=np.float64)
    for i, p in enumerate(prototypes):
        likelihood[i] = gaussian(weight_mv.distance(p))
    likelihood /= likelihood.sum() if likelihood.sum() else 1.0
    posterior = bayesian_update(prior, likelihood)

    # Prototype contributions
    proto_vals = np.array([float(np.dot(p.components, x)) for p in prototypes])
    exp_proto = float(np.dot(posterior, proto_vals))

    # Entropy‑based mixing coefficient
    eps = 1e-12
    entropy = -np.sum(posterior * np.log(posterior + eps))
    alpha = min(1.0, entropy)  # entropy is ≤ log(N); clamp to [0,1]

    return (1.0 - alpha) * lin + alpha * exp_proto


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed reproducibility
    random.seed(42)
    np.random.seed(42)

    # Synthetic NLMS data
    dim = 8
    w = np.random.randn(dim)
    x = np.random.randn(dim)
    target = 0.5  # arbitrary scalar target

    # Bayesian prototypes (5 multivectors of same dimension)
    prototypes = [Multivector(np.random.randn(dim)) for _ in range(5)]
    prior = np.full(5, 1.0 / 5)  # uniform prior

    # Perform a hybrid step
    w_new, posterior = hybrid_nlms_bayes_step(w, x, target, prior, prototypes)

    # Show results
    print("Old weights:", w.round(3))
    print("New weights:", w_new.round(3))
    print("Posterior:", posterior.round(3))

    # Hybrid prediction example
    y = hybrid_predict(w_new, x, posterior, prototypes)
    print("Hybrid prediction:", round(y, 4))

    # Construct RBF graph from the new weights (demonstrates function 3)
    graph = construct_rbf_graph(w_new)
    # Print a tiny excerpt of the graph
    for nid, edges in list(graph.items())[:3]:
        print(f"Node {nid} edges (first 3):", edges[:3])