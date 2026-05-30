# DARWIN HAMMER — match 5805, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1666_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m2673_s3.py (gen5)
# born: 2026-05-30T00:04:46Z

"""Hybrid Algorithm combining:
- Parent A: Hybrid Pheromone‑Tree Bayesian Algorithm
- Parent B: Hybrid Doomsday‑Gini‑Entropy Pheromone utilities

Mathematical Bridge
------------------
Parent A updates edge priors using a Bayesian rule where the likelihood is
derived from the Hamming similarity of 64‑bit perceptual hashes of pheromone
signals. Parent B supplies a Shannon entropy term that quantifies the
uncertainty of the raw pheromone evidence and a Gini coefficient that
measures inequality of the same evidence.

The fusion proceeds as follows:

1. Convert each node’s pheromone vector to a 64‑bit perceptual hash
   (Parent A).
2. Compute the normalized Hamming similarity `S` between two node hashes.
3. Build a likelihood `L` that blends the similarity with the entropy `H`
   of the concatenated evidence:
       L = S * exp(-H)
   (entropy term taken from Parent B).
4. Perform a Bayesian marginal update using `L` as the likelihood and a
   Gini‑derived false‑positive rate `fp = gini(evidence)`.
5. The resulting posterior weight modulates the Euclidean material cost
   of the edge, while a temporal factor based on the Doomsday weekday
   further perturbs the final hybrid cost.

The functions below implement this pipeline and expose a compact
`hybrid_tree_cost` routine that evaluates a weighted tree using the
combined mathematics of both parents."""


import math
import random
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent A utilities (pheromone → hash, Bayesian update)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    # Use up to the first 64 values; pad with zeros if fewer.
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    bits <<= max(0, 64 - len(values))
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute the Bayesian marginal (posterior) probability:
        P(E) = P(E|H)·P(H) + P(E|¬H)·P(¬H)
    where P(E|¬H) is approximated by `false_positive`.
    """
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError('prior, likelihood, and false_positive must be in [0, 1]')
    return likelihood * prior + false_positive * (1.0 - prior)


# ----------------------------------------------------------------------
# Parent B utilities (entropy, Gini, Doomsday, Euclidean)
# ----------------------------------------------------------------------
def shannon_entropy(probs: Iterable[float]) -> float:
    """Shannon entropy (natural log) of a probability‑like distribution."""
    probs_arr = np.array(list(probs), dtype=float)
    probs_arr = probs_arr[probs_arr > 0]  # ignore zeros
    if probs_arr.size == 0:
        return 0.0
    return -float(np.sum(probs_arr * np.log(probs_arr)))


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative vector."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def doomsday_weekday(year: int, month: int, day: int) -> int:
    """
    Return weekday index (0 = Monday … 6 = Sunday) using the Doomsday algorithm.
    The implementation relies on Python's datetime for correctness.
    """
    return (date(year, month, day).weekday() + 1) % 7


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def edge_posterior_weight(
    a_vals: List[float],
    b_vals: List[float],
    prior: float = 0.5,
    false_positive: float = 0.01,
) -> float:
    """
    Compute the Bayesian posterior weight for an edge connecting two nodes.

    Steps:
    1. Hash each pheromone vector (Parent A).
    2. Normalised Hamming similarity S ∈ [0,1].
    3. Concatenate the raw evidence, compute Shannon entropy H (Parent B).
    4. Likelihood L = S * exp(-H).
    5. Posterior = bayes_marginal(prior, L, false_positive).
    """
    # 1. Hashes
    ha = compute_phash(a_vals)
    hb = compute_phash(b_vals)

    # 2. Normalised similarity
    ham = hamming_distance(ha, hb)
    similarity = 1.0 - ham / 64.0

    # 3. Entropy of combined evidence
    combined = a_vals + b_vals
    # Normalise to a probability‑like vector for entropy
    total = sum(combined) if combined else 1.0
    probs = [v / total for v in combined]
    entropy = shannon_entropy(probs)

    # 4. Likelihood
    likelihood = similarity * math.exp(-entropy)

    # 5. Posterior
    posterior = bayes_marginal(prior, likelihood, false_positive)
    return posterior


def hybrid_edge_cost(
    a_pos: Point,
    b_pos: Point,
    a_vals: List[float],
    b_vals: List[float],
    prior: float = 0.5,
    false_positive: float = 0.01,
) -> float:
    """
    Hybrid cost for a single edge.

    The cost blends three ingredients:
    * Material cost – Euclidean distance.
    * Evidence cost – (1 - posterior weight) from the Bayesian update.
    * Inequality & temporal modifiers – Gini coefficient of the evidence and
      a Doomsday weekday factor (weekday/6).

    Returns a non‑negative float.
    """
    # Material distance
    dist = euclidean_length(a_pos, b_pos)

    # Bayesian posterior weight
    posterior = edge_posterior_weight(a_vals, b_vals, prior, false_positive)

    # Gini coefficient of the combined pheromone evidence
    gini = gini_coefficient(a_vals + b_vals)

    # Temporal factor based on today's date
    today = datetime.now(timezone.utc).date()
    weekday_factor = doomsday_weekday(today.year, today.month, today.day) / 6.0  # ∈ [0,1]

    # Assemble hybrid cost
    cost = dist * (1.0 - posterior) * (1.0 + gini) * (1.0 + weekday_factor)
    return max(cost, 0.0)


def hybrid_tree_cost(
    edges: List[Edge],
    node_positions: Dict[str, Point],
    node_pheromones: Dict[str, List[float]],
    prior: float = 0.5,
    false_positive: float = 0.01,
) -> float:
    """
    Compute the total hybrid cost of a tree (or any undirected graph).

    Parameters
    ----------
    edges : list of (node_id_a, node_id_b)
    node_positions : mapping node_id → (x, y)
    node_pheromones : mapping node_id → list of pheromone signal values

    Returns
    -------
    total_cost : float
    """
    total = 0.0
    for u, v in edges:
        pos_u = node_positions[u]
        pos_v = node_positions[v]
        vals_u = node_pheromones[u]
        vals_v = node_pheromones[v]
        total += hybrid_edge_cost(
            pos_u, pos_v, vals_u, vals_v, prior=prior, false_positive=false_positive
        )
    return total


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple three‑node tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, math.sqrt(3) / 2),
    }

    # Random pheromone vectors (positive floats)
    random.seed(42)
    pheromones = {
        nid: [random.random() for _ in range(20)] for nid in nodes
    }

    # Edges forming a triangle (tree would drop one edge, but we keep all for demo)
    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    cost = hybrid_tree_cost(edges, nodes, pheromones)
    print(f"Hybrid tree cost (demo): {cost:.6f}")