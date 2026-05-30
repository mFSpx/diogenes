# DARWIN HAMMER — match 5427, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1666_s1.py (gen5)
# parent_b: hybrid_hybrid_regret_engine_hybrid_hybrid_bandit_m83_s1.py (gen3)
# born: 2026-05-30T00:01:47Z

"""Hybrid Pheromone‑Regret‑Koopman Algorithm
================================================

Parent algorithms:
- **A**: `hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1666_s1.py`
  (pheromone‑based perceptual hashing, Hamming similarity, Bayesian edge update).
- **B**: `hybrid_hybrid_regret_engine_hybrid_hybrid_bandit_m83_s1.py`
  (regret‑weighted strategy computation and Koopman‑operator evolution).

Mathematical bridge
------------------
The bridge is built on the *edge likelihood* produced by algorithm A.
For any two nodes we compute a 64‑bit perceptual hash from their feature
vectors, obtain a Hamming similarity `s ∈ [0,1]` and treat `s` as a
likelihood `L(e)` that the edge `e` is relevant.  
This likelihood is used as a **Bayesian update** of a uniform prior
`P₀(e)=1/|E|`, yielding a posterior edge weight `w(e)`.  

The posterior weights modulate the *expected value* of actions in the
regret engine of algorithm B:

Ê_i = (expected_value_i – cost_i – risk_i + cf_i) * w(e_i)

where `e_i` is the edge traversed by action `i` and `cf_i` is the
counter‑factual contribution.

The resulting regret‑weighted distribution is then propagated forward
by a **Koopman operator** `K` (a linear matrix) producing the next‑step
strategy vector.  This yields a single unified system that adapts
routing decisions using pheromone evidence, regret minimisation, and
linear dynamical evolution.

The module below implements this hybrid pipeline.
"""

from __future__ import annotations
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Algorithm A – pheromone / hashing utilities
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    # Pad remaining bits with zeros if fewer than 64 values
    bits <<= max(0, 64 - len(values))
    return bits & ((1 << 64) - 1)


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def pheromone_likelihood(feat_a: List[float], feat_b: List[float]) -> float:
    """
    Compute a Bayesian‑compatible likelihood for an edge between two nodes.
    The likelihood is the normalized Hamming similarity of their perceptual
    hashes (value in [0,1]).
    """
    ha = compute_phash(feat_a)
    hb = compute_phash(feat_b)
    dist = hamming_distance(ha, hb)
    similarity = 1.0 - dist / 64.0
    # Clamp to avoid exact 0 or 1 which could break later normalisation
    return max(1e-6, min(1.0 - 1e-6, similarity))


def posterior_edge_weights(
    edges: List[Edge],
    node_features: Dict[str, List[float]],
) -> Dict[Edge, float]:
    """
    Bayesian update of a uniform prior using pheromone likelihoods.
    Returns a dictionary mapping each edge to its posterior weight.
    """
    if not edges:
        return {}
    # Uniform prior
    prior = 1.0 / len(edges)
    # Compute unnormalised posteriors
    unnorm: Dict[Edge, float] = {}
    for e in edges:
        f_a = node_features.get(e[0], [])
        f_b = node_features.get(e[1], [])
        lik = pheromone_likelihood(f_a, f_b)
        unnorm[e] = lik * prior
    total = sum(unnorm.values()) or 1.0
    return {e: v / total for e, v in unnorm.items()}


# ----------------------------------------------------------------------
# Algorithm B – regret engine utilities
# ----------------------------------------------------------------------
class MathAction:
    __slots__ = ("id", "expected_value", "cost", "risk")
    def __init__(self, id: str, expected_value: float, cost: float = 0.0, risk: float = 0.0):
        self.id = id
        self.expected_value = expected_value
        self.cost = cost
        self.risk = risk


class MathCounterfactual:
    __slots__ = ("action_id", "outcome_value", "probability")
    def __init__(self, action_id: str, outcome_value: float, probability: float = 1.0):
        self.action_id = action_id
        self.outcome_value = outcome_value
        self.probability = probability


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Classic regret‑weighted softmax distribution over actions.
    """
    if not actions:
        return {}
    cf_map = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf_map.get(a.id, 0.0)
        for a in actions
    }
    best = max(vals.values())
    # Use log‑sum‑exp style stability
    exp_vals = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(exp_vals.values()) or 1.0
    return {k: v / total for k, v in exp_vals.items()}


# ----------------------------------------------------------------------
# Fusion: hybrid regret strategy modulated by pheromone posteriors
# ----------------------------------------------------------------------
def hybrid_regret_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    edge_map: Dict[str, Edge],
    edge_posteriors: Dict[Edge, float],
) -> Dict[str, float]:
    """
    Adjust each action's expected value by the posterior weight of the edge it
    traverses, then compute the regret‑weighted distribution.
    """
    # Build a temporary list with modified expected values
    mod_actions: List[MathAction] = []
    for a in actions:
        edge = edge_map.get(a.id)
        weight = edge_posteriors.get(edge, 1.0) if edge else 1.0
        # Scale the net value by the posterior weight
        net = (a.expected_value - a.cost - a.risk) * weight
        # Incorporate counterfactuals later in the regret function
        mod_actions.append(MathAction(a.id, net, cost=0.0, risk=0.0))
    return compute_regret_weighted_strategy(mod_actions, counterfactuals)


def koopman_step(
    strategy_vec: np.ndarray,
    koopman_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply a Koopman operator (linear matrix) to a probability vector.
    The result is re‑normalised to stay a proper distribution.
    """
    if koopman_matrix.shape[1] != strategy_vec.shape[0]:
        raise ValueError("Koopman matrix column size must match strategy vector length")
    next_vec = koopman_matrix @ strategy_vec
    # Clip negatives (numerical artefacts) and renormalise
    next_vec = np.clip(next_vec, 0.0, None)
    total = next_vec.sum()
    if total == 0.0:
        # fallback to uniform distribution
        return np.full_like(next_vec, 1.0 / next_vec.size)
    return next_vec / total


def hybrid_pipeline(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    edges: List[Edge],
    node_features: Dict[str, List[float]],
    action_to_edge: Dict[str, Edge],
    koopman_matrix: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Full hybrid operation:
    1. Compute posterior edge weights from pheromone evidence.
    2. Produce a regret‑weighted strategy modulated by those weights.
    3. Convert the distribution to a vector and evolve it with the Koopman operator.
    Returns the intermediate distribution (dict) and the evolved vector.
    """
    posteriors = posterior_edge_weights(edges, node_features)
    regret_dist = hybrid_regret_strategy(actions, counterfactuals, action_to_edge, posteriors)

    # Convert dict to ordered vector (alphabetical action order for reproducibility)
    ordered_ids = sorted(regret_dist.keys())
    vec = np.array([regret_dist[i] for i in ordered_ids], dtype=float)

    evolved_vec = koopman_step(vec, koopman_matrix)
    # Map evolved vector back to dict with same ordering
    evolved_dist = {aid: float(val) for aid, val in zip(ordered_ids, evolved_vec)}
    return evolved_dist, evolved_vec


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy node features (random but deterministic for repeatability)
    random.seed(42)
    node_features = {
        "A": [random.random() for _ in range(10)],
        "B": [random.random() for _ in range(10)],
        "C": [random.random() for _ in range(10)],
        "D": [random.random() for _ in range(10)],
    }

    # Define edges and a simple mapping from actions to edges
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]
    action_to_edge = {
        "a1": ("A", "B"),
        "a2": ("B", "C"),
        "a3": ("C", "D"),
        "a4": ("A", "D"),
    }

    # Create actions (expected_value roughly proportional to Euclidean distance)
    def euclidean(f1: List[float], f2: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(f1, f2)))

    actions = []
    for aid, edge in action_to_edge.items():
        v = euclidean(node_features[edge[0]], node_features[edge[1]])
        actions.append(MathAction(aid, expected_value=v, cost=0.1 * v, risk=0.05 * v))

    # Counterfactuals (empty for this test)
    counterfactuals: List[MathCounterfactual] = []

    # Simple Koopman matrix: slight diffusion (each state keeps 80% mass, spreads 20% uniformly)
    n = len(actions)
    K = np.full((n, n), 0.2 / (n - 1))
    np.fill_diagonal(K, 0.8)

    # Run the pipeline
    final_dist, final_vec = hybrid_pipeline(
        actions,
        counterfactuals,
        edges,
        node_features,
        action_to_edge,
        K,
    )

    print("Posterior‑weighted regret distribution:")
    for k, v in sorted(final_dist.items()):
        print(f"  {k}: {v:.4f}")

    print("\nEvolved vector (sum should be 1.0):")
    print(final_vec)
    print("Sum:", final_vec.sum())