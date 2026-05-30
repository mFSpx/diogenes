# DARWIN HAMMER — match 2673, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py (gen4)
# born: 2026-05-29T23:43:35Z

"""Hybrid Algorithm integrating:
- hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s0.py (Gini coefficient,
  Bayesian update, tree cost)
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py (Shannon entropy,
  pheromone signal hashing, entropy‑driven signal value)

Mathematical bridge:
The pheromone signals recorded on each node are first summarised by their
Shannon entropy **H**.  The entropy is turned into a likelihood
`L = exp(-H)` (high uncertainty → low likelihood).  Independently the
distribution of the raw signal values is characterised by the Gini
coefficient **G**, measuring inequality.  A combined weight

    w = L * (1 - G)

is therefore a Bayesian‑style likelihood that is additionally penalised
by inequality of the underlying signal distribution.  This weight is
used as the likelihood in a Bayesian update of a prior node probability
distribution and also scales the edge costs in the minimum‑cost tree
computation.

The module therefore fuses the two parent topologies into a single
system that can:
1. Quantify signal inequality (Gini) and uncertainty (entropy).
2. Perform a Bayesian update of node priors using the combined weight.
3. Compute a minimum‑cost spanning tree whose edge costs are modulated
   by the same weight, yielding a cost that reflects both signal
   quality and distributional fairness.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Basic utilities from Parent A
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str, float]  # (parent, child, base_weight)


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index (0=Monday … 6=Sunday) using the Doomsday rule."""
    return (datetime(year, month, day).weekday() + 1) % 7


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative sequence."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Utilities from Parent B (entropy & pheromone hashing)
# ----------------------------------------------------------------------
def shannon_entropy(probs: Iterable[float]) -> float:
    """Standard Shannon entropy (base e) for a probability distribution."""
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log(p) for p in probs)


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 64‑bit binary based on mean threshold."""
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


def broadcast_probability(phase: int, step: int) -> float:
    """A smooth decreasing broadcast probability (placeholder implementation)."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    # Example: exponential decay with respect to step, modulated by phase
    return math.exp(-step / (10.0 * phase))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_node_weights(
    signals: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    For each node compute a combined weight w = exp(-H) * (1 - G),
    where H is the Shannon entropy of the normalised signal distribution
    and G is the Gini coefficient of the raw signals.
    """
    weights: Dict[str, float] = {}
    for node, vals in signals.items():
        if not vals:
            weights[node] = 0.0
            continue

        # Normalise to a probability distribution for entropy
        total = sum(vals)
        probs = [v / total for v in vals]

        H = shannon_entropy(probs)
        G = gini_coefficient(vals)

        likelihood = math.exp(-H)          # high entropy → low likelihood
        inequality_penalty = 1.0 - G       # high inequality → lower weight
        weights[node] = likelihood * inequality_penalty
    return weights


def bayesian_update(
    prior: Dict[str, float],
    likelihoods: Dict[str, float]
) -> Dict[str, float]:
    """
    Perform a Bayesian update of node priors using the supplied likelihoods.
    posterior_i ∝ prior_i * likelihood_i
    """
    unnorm = {k: prior.get(k, 0.0) * likelihoods.get(k, 0.0) for k in prior}
    total = sum(unnorm.values())
    if total == 0:
        # Avoid division by zero – return uniform distribution
        n = len(prior)
        return {k: 1.0 / n for k in prior}
    return {k: v / total for k, v in unnorm.items()}


def minimum_cost_tree(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    node_weights: Dict[str, float],
    path_weight: float = 0.2,
) -> float:
    """
    Compute the cost of a minimum‑spanning tree rooted at *root*.
    Each edge cost is scaled by the average weight of its two incident nodes
    and by a fixed *path_weight* factor.
    """
    # Build adjacency with dynamic costs
    adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes}
    for u, v, base in edges:
        w_u = node_weights.get(u, 1.0)
        w_v = node_weights.get(v, 1.0)
        scale = (w_u + w_v) / 2.0 * path_weight
        cost = base * scale
        adj[u].append((v, cost))
        adj[v].append((u, cost))

    # Prim's algorithm for MST (undirected)
    visited = {root}
    edge_candidates: List[Tuple[float, str, str]] = []
    for nbr, c in adj[root]:
        edge_candidates.append((c, root, nbr))
    total_cost = 0.0

    import heapq
    heapq.heapify(edge_candidates)

    while edge_candidates and len(visited) < len(nodes):
        cost, src, dst = heapq.heappop(edge_candidates)
        if dst in visited:
            continue
        visited.add(dst)
        total_cost += cost
        for nbr, c in adj[dst]:
            if nbr not in visited:
                heapq.heappush(edge_candidates, (c, dst, nbr))

    return total_cost


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def generate_random_pheromone_signals(
    node_names: List[str],
    seed: int = 42,
    min_val: float = 0.0,
    max_val: float = 10.0,
    count: int = 20,
) -> Dict[str, List[float]]:
    """Create synthetic pheromone signal lists for each node."""
    rnd = random.Random(seed)
    signals: Dict[str, List[float]] = {}
    for name in node_names:
        signals[name] = [rnd.uniform(min_val, max_val) for _ in range(count)]
    return signals


def example_hybrid_workflow() -> Tuple[Dict[str, float], float]:
    """
    Execute a full hybrid pipeline:
    1. Generate synthetic signals.
    2. Compute node weights (entropy + Gini).
    3. Update a uniform prior via Bayesian update.
    4. Compute the minimum‑cost tree cost.
    Returns the posterior distribution and the tree cost.
    """
    # 1. Nodes and geometry
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }

    # 2. Edges with Euclidean base cost
    edges: List[Edge] = [
        ("A", "B", length(nodes["A"], nodes["B"])),
        ("A", "C", length(nodes["A"], nodes["C"])),
        ("B", "D", length(nodes["B"], nodes["D"])),
        ("C", "D", length(nodes["C"], nodes["D"])),
        ("B", "C", length(nodes["B"], nodes["C"])),
    ]

    # 3. Synthetic pheromone signals
    signals = generate_random_pheromone_signals(list(nodes.keys()))

    # 4. Compute combined node weights
    weights = compute_node_weights(signals)

    # 5. Uniform prior over nodes
    prior = {n: 1.0 / len(nodes) for n in nodes}

    # 6. Bayesian update using the weights as likelihoods
    posterior = bayesian_update(prior, weights)

    # 7. Minimum‑cost tree cost using posterior‑derived weights
    cost = minimum_cost_tree(nodes, edges, root="A", node_weights=weights)

    return posterior, cost


def hash_based_similarity(
    signals_a: List[float],
    signals_b: List[float],
) -> float:
    """
    Compute similarity between two pheromone signal sets using perceptual hash
    and Hamming distance.  Returns a value in [0, 1] where 1 means identical.
    """
    hash_a = compute_phash(signals_a)
    hash_b = compute_phash(signals_b)
    max_bits = max(hash_a.bit_length(), hash_b.bit_length(), 1)
    ham = hamming_distance(hash_a, hash_b)
    return 1.0 - ham / max_bits


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    posterior_dist, tree_cost = example_hybrid_workflow()
    print("Posterior distribution after hybrid Bayesian update:")
    for node, prob in sorted(posterior_dist.items()):
        print(f"  {node}: {prob:.4f}")

    print(f"\nMinimum‑cost tree total cost: {tree_cost:.4f}")

    # Demonstrate hash similarity on two random signal vectors
    sig1 = [random.random() for _ in range(30)]
    sig2 = [random.random() for _ in range(30)]
    sim = hash_based_similarity(sig1, sig2)
    print(f"\nHash‑based similarity between two random signal sets: {sim:.3f}")

    # Verify that broadcast_probability runs without error
    print("\nSample broadcast probabilities:")
    for phase in range(1, 4):
        for step in [1, 5, 10]:
            print(f"  phase={phase}, step={step} -> {broadcast_probability(phase, step):.4f}")