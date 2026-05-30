# DARWIN HAMMER — match 5096, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_minimum_cost__hybrid_sketches_hybr_m1924_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1426_s2.py (gen5)
# born: 2026-05-29T23:59:52Z

"""Hybrid Minimum‑Cost Bayesian‑Sketch Tree with Clifford‑Geometric Edge Fusion

Parents
-------
* **Parent A** – ``hybrid_hybrid_minimum_cost__hybrid_sketches_hybr_m1924_s0.py``
  Provides Euclidean edge lengths, Bayesian posterior relevance weights
  ``w_B(e)`` and a count‑min sketch estimate ``ĉ(e)`` of how often an edge
  has been selected by a routing policy.

* **Parent B** – ``hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1426_s2.py``
  Supplies a minimal Clifford algebra ``Cl(3,0)`` (grade‑1 vectors only) and
  the geometric product whose scalar part is the ordinary dot product.
  This allows a similarity measure between multivectors.

Mathematical Bridge
-------------------
For each edge ``e`` we build a 3‑dimensional grade‑1 multivector

    m(e) = ( ℓ(e) ,  w_B(e) ,  log(1 + ĉ(e)) )   ∈  Cl(3,0)

where ``ℓ(e)`` is the Euclidean length, ``w_B(e)`` the Bayesian posterior,
and ``log(1 + ĉ(e))`` the exploration term from the sketch.
The scalar part of the geometric product of two such multivectors is the
dot product:

    ⟨ m(e₁) * m(e₂) ⟩₀ = ℓ(e₁)·ℓ(e₂) + w_B(e₁)·w_B(e₂) + log₁·log₂

We use this similarity to bias a bandit‑style edge selection:
the reward for edge ``e`` is

    R(e) = -[ ℓ(e) + α·w_B(e) + β·log(1 + ĉ(e)) ] + γ·⟨ m(e) * m_target ⟩₀

with tunable coefficients ``α,β,γ > 0`` and a target multivector
``m_target`` that encodes the desired routing characteristics.
The algorithm therefore fuses the spatial, probabilistic and
exploratory topologies of Parent A with the algebraic similarity
machinery of Parent B into a single unified decision system.
"""

import math
import random
import sys
import pathlib
import hashlib
from typing import Tuple, List, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian update utilities (Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) for a Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability P(H|E) given prior, likelihood and marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Count‑Min Sketch utilities (Parent B – simplified)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min Sketch with pairwise‑independent hash functions."""
    def __init__(self, width: int = 1000, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(0, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: Any, i: int) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(str(item).encode('utf-8'))
        h.update(self.seeds[i].to_bytes(4, 'little'))
        return int.from_bytes(h.digest(), 'little') % self.width

    def update(self, item: Any, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += increment

    def estimate(self, item: Any) -> int:
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))

# ----------------------------------------------------------------------
# Clifford algebra (grade‑1 only) – geometric product scalar part (Parent B)
# ----------------------------------------------------------------------
def geometric_scalar_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Scalar part of the geometric product of two grade‑1 multivectors.
    For vectors this is simply the Euclidean dot product.
    """
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same dimension")
    return float(np.dot(v1, v2))

# ----------------------------------------------------------------------
# Hybrid edge representation
# ----------------------------------------------------------------------
def edge_features(
    edge: Edge,
    points: Dict[str, Point],
    prior: float,
    likelihood: float,
    false_positive: float,
    sketch: CountMinSketch,
    alpha: float,
    beta: float,
) -> Tuple[float, float, float]:
    """
    Compute the three core features of an edge:

    * ℓ(e)          – Euclidean length
    * w_B(e)       – Bayesian posterior relevance
    * log(1 + ĉ)   – Exploration term from the sketch
    """
    a, b = edge
    ℓ = length(points[a], points[b])
    marginal = bayes_marginal(prior, likelihood, false_positive)
    w_B = bayes_update(prior, likelihood, marginal)
    count_est = sketch.estimate(edge)
    log_cnt = math.log1p(count_est)  # log(1 + ĉ(e))
    return ℓ, w_B, log_cnt

def edge_to_multivector(features: Tuple[float, float, float]) -> np.ndarray:
    """
    Map edge features to a grade‑1 multivector in Cl(3,0).
    The vector is returned as a NumPy array of shape (3,).
    """
    ℓ, w_B, log_cnt = features
    return np.array([ℓ, w_B, log_cnt], dtype=np.float64)

def hybrid_edge_cost(
    features: Tuple[float, float, float],
    alpha: float,
    beta: float,
) -> float:
    """
    Base hybrid cost C(e) = ℓ + α·w_B + β·log(1 + ĉ).
    The features tuple already contains ℓ, w_B and log(1 + ĉ).
    """
    ℓ, w_B, log_cnt = features
    return ℓ + alpha * w_B + beta * log_cnt

# ----------------------------------------------------------------------
# Bandit‑style edge selection using the fused metric
# ----------------------------------------------------------------------
def select_edge_bandit(
    edges: List[Edge],
    points: Dict[str, Point],
    priors: Dict[Edge, float],
    likelihoods: Dict[Edge, float],
    false_positive: float,
    sketch: CountMinSketch,
    alpha: float,
    beta: float,
    gamma: float,
    target_mv: np.ndarray,
) -> Edge:
    """
    Compute a probability distribution over edges and sample one.

    Reward for edge e:
        R(e) = - hybrid_edge_cost(e) + γ * ⟨ m(e) * target_mv ⟩₀

    where m(e) is the multivector built from the edge's three features.
    The distribution is the soft‑max of the rewards.
    """
    rewards: List[float] = []
    for e in edges:
        prior = priors.get(e, 0.5)          # default prior if missing
        likelihood = likelihoods.get(e, 0.5)
        feats = edge_features(e, points, prior, likelihood, false_positive, sketch, alpha, beta)
        cost = hybrid_edge_cost(feats, alpha, beta)
        mv = edge_to_multivector(feats)
        similarity = geometric_scalar_similarity(mv, target_mv)
        reward = -cost + gamma * similarity
        rewards.append(reward)

    # Numerical stability for soft‑max
    max_r = max(rewards)
    exp_vals = [math.exp(r - max_r) for r in rewards]
    total = sum(exp_vals)
    probs = [v / total for v in exp_vals]

    # Sample according to the computed probabilities
    chosen = random.choices(edges, weights=probs, k=1)[0]
    # Update sketch to reflect the selection (exploration tracking)
    sketch.update(chosen, increment=1)
    return chosen

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    # Define a tiny graph
    points = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "C")]

    # Prior and likelihood per edge (arbitrary for the test)
    priors = {e: 0.6 for e in edges}
    likelihoods = {e: 0.7 for e in edges}
    false_positive = 0.1

    # Sketch instance
    sketch = CountMinSketch(width=200, depth=4, seed=42)

    # Hyper‑parameters
    alpha = 0.5
    beta = 0.3
    gamma = 0.8

    # Target multivector – we desire short, highly relevant, rarely used edges
    target_features = (0.0, 1.0, 0.0)   # length≈0, high w_B, low log‑count
    target_mv = np.array(target_features, dtype=np.float64)

    # Run a few selection steps
    for step in range(5):
        chosen = select_edge_bandit(
            edges,
            points,
            priors,
            likelihoods,
            false_positive,
            sketch,
            alpha,
            beta,
            gamma,
            target_mv,
        )
        print(f"Step {step+1}: selected edge {chosen}")

if __name__ == "__main__":
    _smoke_test()