# DARWIN HAMMER — match 954, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s4.py (gen4)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s2.py (gen3)
# born: 2026-05-29T23:31:50Z

"""
Hybrid Algorithm: Semantic–Geometric Regret‑Aware Hoeffding Engine
================================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s4.py``  
  Provides probabilistic primitives (broadcast probability, Hoeffding bound) and
  a regret‑based decision model (``MathAction`` / ``MathCounterfactual``).

* **Parent B** – ``hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s2.py``  
  Provides semantic similarity, pheromone‑based probabilities, entropy, a simple
  Voronoi partitioning and a geometric (Clifford) product between a semantic
  vector **v** and a pheromone vector **π**.

Mathematical Bridge
-------------------
The bridge is the *regret* scalar that modulates both statistical confidence
(Hoeffding bound) and geometric interaction.  Regret is interpreted as an
“energy” that scales the Hoeffding confidence interval and also weights the
scalar part of the geometric product (the dot product **v·π**).  The bivector
part of the product encodes orientation‑weighted interaction and is combined
with the entropy of the pheromone distribution to produce an exploration term.
Finally, a tropical *max‑plus* aggregation fuses the confidence‑adjusted scalar
and the entropy‑adjusted bivector into a single similarity score.

The core hybrid functions are:

1. ``regret_aware_hoeffding`` – Hoeffding bound scaled by regret.
2. ``geometric_pheromone_product`` – Clifford product of semantic and pheromone
   vectors, returning scalar, bivector norm and an entropy‑adjusted exploration
   term.
3. ``hybrid_similarity`` – Tropical max‑plus combination of the two components
   to yield a unified similarity measure usable in neighbourhood or routing
   decisions.

All functions rely only on ``numpy`` and the Python standard library.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Mapping, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Types shared across the hybrid
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]
Vector = np.ndarray

# ----------------------------------------------------------------------
# Parent A – probabilistic / regret primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def compute_hoeffding_bound(observed_gain: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound for a bounded gain."""
    if n <= 0:
        raise ValueError('sample size n must be positive')
    return math.sqrt((observed_gain * math.log(2 / delta)) / (2 * n))


@dataclass(frozen=True)
class MathAction:
    """Atomic action used by the regret engine."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Outcome that could have happened for a given action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def regret_aware_hoeffding(
    observed_gain: float,
    delta: float,
    n: int,
    regret: float,
) -> float:
    """
    Hoeffding bound scaled by a regret factor.

    The regret is interpreted as an energy that inflates the confidence
    interval: larger regret → looser bound (more uncertainty).

    Returns:
        float: regret‑aware bound.
    """
    base = compute_hoeffding_bound(observed_gain, delta, n)
    # Regret is non‑negative; we map it to a multiplicative factor > 1
    factor = 1.0 + max(0.0, regret)
    return base * factor


# ----------------------------------------------------------------------
# Parent B – semantic / geometric primitives
# ----------------------------------------------------------------------
def _cos(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.fromiter(a, dtype=float)
    b_arr = np.fromiter(b, dtype=float)
    den = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return 0.0 if den == 0 else float(np.dot(a_arr, b_arr) / den)


def pheromone_probabilities(pheromones: Iterable[float]) -> List[float]:
    """Normalize a pheromone vector into a probability distribution."""
    pher = list(pheromones)
    total = sum(pher)
    if total == 0:
        raise ValueError("pheromone vector must contain positive mass")
    return [p / total for p in pher]


def entropy(probabilities: Iterable[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    probs = np.array(list(probabilities), dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = probs / total
    probs = np.clip(probs, eps, None)
    return -float(np.sum(probs * np.log(probs)))


def assign_voronoi(points: List[Vector], centroids: List[Vector]) -> dict[int, int]:
    """
    Very simple Voronoi assignment: each point is assigned to the nearest centroid.

    Returns:
        dict mapping point index -> centroid index (cell id).
    """
    if not points:
        return {}
    if not centroids:
        raise ValueError("centroids list cannot be empty")
    assignment = {}
    for i, p in enumerate(points):
        distances = [np.linalg.norm(p - c) for c in centroids]
        assignment[i] = int(np.argmin(distances))
    return assignment


def geometric_pheromone_product(
    semantic_vec: Vector,
    pheromone_vec: Vector,
) -> Tuple[float, float, float]:
    """
    Compute the Clifford (geometric) product between a semantic vector **v**
    and a pheromone vector **π**.

    Returns:
        scalar_part   – dot product v·π (grade‑0 component)
        bivector_norm – Frobenius norm of the antisymmetric part (grade‑2 component)
        exploration   – entropy‑adjusted exploration term derived from π
    """
    if semantic_vec.shape != pheromone_vec.shape:
        raise ValueError("vectors must have the same dimensionality")

    # Grade‑0 (scalar) part
    scalar = float(np.dot(semantic_vec, pheromone_vec))

    # Grade‑2 (bivector) part: antisymmetric outer product
    outer = np.outer(semantic_vec, pheromone_vec)
    bivector = outer - outer.T
    bivector_norm = float(np.linalg.norm(bivector, ord='fro'))

    # Exploration term from pheromone entropy
    probs = pheromone_probabilities(pheromone_vec)
    ent = entropy(probs)
    exploration = bivector_norm * (1.0 + ent)

    return scalar, bivector_norm, exploration


# ----------------------------------------------------------------------
# Hybrid Functions (the required three)
# ----------------------------------------------------------------------
def hybrid_regret_hoeffding_score(
    observed_gain: float,
    delta: float,
    n: int,
    regret: float,
    weight: float = 0.5,
) -> float:
    """
    Combine a regret‑aware Hoeffding bound with a simple weight to obtain a
    confidence score.  The score is lower when the bound is tight (high confidence)
    and higher when regret is large.

    Args:
        observed_gain: empirical gain observed for an action.
        delta: confidence level (e.g., 0.05).
        n: number of observations.
        regret: regret value associated with the action.
        weight: blending factor between bound and regret (0..1).

    Returns:
        float: blended confidence score.
    """
    bound = regret_aware_hoeffding(observed_gain, delta, n, regret)
    # Normalise bound to [0,1] using a simple sigmoid for stability
    norm_bound = 1.0 / (1.0 + math.exp(-bound))
    # Regret is already non‑negative; map to [0,1] with a logistic transform
    norm_regret = 1.0 / (1.0 + math.exp(-regret))
    return weight * norm_bound + (1.0 - weight) * norm_regret


def hybrid_geometric_score(
    semantic_vec: Vector,
    pheromone_vec: Vector,
    regret: float,
    alpha: float = 0.7,
) -> float:
    """
    Produce a geometric similarity score that is modulated by regret.

    The scalar (dot) part is multiplied by (1 + regret) to reward low‑regret
    configurations, while the bivector‑derived exploration term is blended with
    the scalar via a tropical max‑plus operation.

    Args:
        semantic_vec: semantic embedding of the item.
        pheromone_vec: pheromone mass vector attached to the same item.
        regret: regret value influencing the scalar part.
        alpha: weight for the max‑plus aggregation (0..1).

    Returns:
        float: hybrid similarity score.
    """
    scalar, biv_norm, exploration = geometric_pheromone_product(semantic_vec, pheromone_vec)

    # Regret‑aware scalar
    reg_scalar = scalar * (1.0 + max(0.0, regret))

    # Tropical max‑plus: max(a + w, b + (1-w))
    # Here we treat reg_scalar and exploration as the two terms.
    max_plus = max(reg_scalar + alpha, exploration + (1.0 - alpha))
    return max_plus


def hybrid_similarity(
    node_vec: Vector,
    pheromone_vec: Vector,
    observed_gain: float,
    delta: float,
    n: int,
    regret: float,
    weight_conf: float = 0.5,
    alpha_geo: float = 0.7,
) -> float:
    """
    Full hybrid similarity that fuses:

    * Regret‑aware Hoeffding confidence (statistical reliability)
    * Geometric pheromone interaction (semantic‑spatial structure)
    * Tropical max‑plus aggregation (non‑linear fusion)

    The result can be used to rank neighbours, select actions, or drive
    broadcast decisions in a distributed system.

    Returns:
        float: final similarity / utility score.
    """
    conf_score = hybrid_regret_hoeffding_score(
        observed_gain, delta, n, regret, weight=weight_conf
    )
    geo_score = hybrid_geometric_score(node_vec, pheromone_vec, regret, alpha=alpha_geo)

    # Tropical max‑plus of the two high‑level scores
    final_score = max(conf_score + weight_conf, geo_score + (1.0 - weight_conf))
    return final_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic data
    dim = 5
    rng = np.random.default_rng(42)

    # Semantic vectors for three nodes
    semantic_vectors = [rng.normal(size=dim) for _ in range(3)]

    # Pheromone vectors (non‑negative)
    pheromone_vectors = [np.abs(rng.normal(size=dim)) + 0.1 for _ in range(3)]

    # Random regret and observed gains
    regrets = [abs(rng.normal()) for _ in range(3)]
    gains = [abs(rng.normal()) for _ in range(3)]

    # Parameters for Hoeffding bound
    delta = 0.05
    n_samples = 30

    # Compute hybrid scores
    for i in range(3):
        score = hybrid_similarity(
            node_vec=semantic_vectors[i],
            pheromone_vec=pheromone_vectors[i],
            observed_gain=gains[i],
            delta=delta,
            n=n_samples,
            regret=regrets[i],
        )
        print(f"Node {i} – Hybrid Score: {score:.4f}")

    # Demonstrate Voronoi assignment (using the same semantic vectors as points)
    centroids = [rng.normal(size=dim) for _ in range(2)]
    assignment = assign_voronoi(semantic_vectors, centroids)
    print("Voronoi cell assignment:", assignment)