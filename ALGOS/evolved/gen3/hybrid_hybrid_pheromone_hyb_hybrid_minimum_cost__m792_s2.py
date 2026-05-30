# DARWIN HAMMER — match 792, survivor 2
# gen: 3
# parent_a: hybrid_pheromone_hybrid_distributed_l_m41_s0.py (gen2)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:30:55Z

"""Hybrid Pheromone‑Tree Bayesian Algorithm
Parents:
- hybrid_pheromone_hybrid_distributed_l_m41_s0.py (perceptual hashing of pheromone signals)
- hybrid_minimum_cost_tree_bayes_update_m6_s1.py (minimum‑cost tree with Bayesian edge updates)

Mathematical Bridge:
Node‑wise pheromone signal vectors are reduced to 64‑bit perceptual hashes.
The Hamming similarity between two node hashes provides a data‑driven likelihood
that an edge between those nodes is “relevant”.  This likelihood is fed into a
Bayesian update of the edge prior probability.  The resulting posterior edge
weights modulate the material cost in the tree‑cost function, yielding a hybrid
cost that accounts for both physical distance and pheromone‑based evidence.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Type aliases
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Perceptual hashing utilities (from Parent A)
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
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    """Probability that a pheromone broadcast succeeds at a given phase/step."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

# ----------------------------------------------------------------------
# Bayesian utilities (from Parent B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def canonical_edge(e: Edge) -> Edge:
    """Return a deterministic ordering for an undirected edge."""
    return tuple(sorted(e))

def update_edge_priors(
    node_signals: Dict[str, List[float]],
    initial_priors: Dict[Edge, float],
    false_positive: float = 0.01,
) -> Dict[Edge, float]:
    """
    Compute posterior edge priors using perceptual‑hash similarity as likelihood.

    Args:
        node_signals: mapping node_id -> list of pheromone signal values.
        initial_priors: mapping undirected edge -> prior probability (0..1).
        false_positive: assumed false‑positive rate for the evidence.

    Returns:
        New dict with the same keys as ``initial_priors`` containing posterior priors.
    """
    # Compute a hash per node
    node_hashes: Dict[str, int] = {
        nid: compute_phash(vals) for nid, vals in node_signals.items()
    }

    posteriors: Dict[Edge, float] = {}
    for edge, prior in initial_priors.items():
        u, v = edge
        h_u, h_v = node_hashes.get(u, 0), node_hashes.get(v, 0)
        # Similarity in [0,1]
        ham = hamming_distance(h_u, h_v)
        similarity = 1.0 - ham / 64.0
        # Use similarity as the likelihood that the edge is truly relevant
        likelihood = similarity
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        posteriors[canonical_edge(edge)] = posterior
    return posteriors

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    path_weight: float = 0.2,
) -> float:
    """
    Minimum‑cost tree cost where each edge length is weighted by its posterior prior.

    The cost = Σ (edge_length * posterior) + path_weight * Σ(distances from root).
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        prior = edge_priors.get(canonical_edge((a, b)), 0.5)  # fallback to 0.5 if missing
        material += length(nodes[a], nodes[b]) * prior

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    visited = set([root])
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    return material + path_weight * sum(dist.values())

def hybrid_broadcast_phase(
    nodes: Dict[str, List[float]],
    edges: List[Edge],
    phase: int,
    step: int,
) -> Dict[Edge, bool]:
    """
    Simulate a broadcast phase where each edge succeeds with probability
    given by ``broadcast_probability`` scaled by the pheromone similarity.

    Returns a dict mapping edge -> bool indicating successful transmission.
    """
    prob = broadcast_probability(phase, step)
    successes: Dict[Edge, bool] = {}
    # Use hash similarity to modulate the base probability
    node_hashes = {nid: compute_phash(vals) for nid, vals in nodes.items()}
    for a, b in edges:
        ham = hamming_distance(node_hashes.get(a, 0), node_hashes.get(b, 0))
        similarity = 1.0 - ham / 64.0
        edge_prob = prob * similarity
        successes[canonical_edge((a, b))] = random.random() < edge_prob
    return successes

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    nodes_pos: Dict[str, Point] = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }

    # Synthetic pheromone signal vectors (random floats)
    random.seed(42)
    node_signals: Dict[str, List[float]] = {
        nid: [random.random() for _ in range(20)] for nid in nodes_pos
    }

    edges: List[Edge] = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]

    # Uniform priors
    init_priors: Dict[Edge, float] = {canonical_edge(e): 0.5 for e in edges}

    # Update priors using perceptual hash similarity
    post_priors = update_edge_priors(node_signals, init_priors)

    # Compute hybrid cost
    cost = hybrid_tree_cost(nodes_pos, edges, root="A", edge_priors=post_priors)
    print(f"Hybrid tree cost: {cost:.4f}")

    # Run a broadcast simulation for phase=2, step=1
    successes = hybrid_broadcast_phase(node_signals, edges, phase=2, step=1)
    print("Broadcast successes per edge:")
    for e, ok in successes.items():
        print(f"  {e}: {'OK' if ok else 'FAIL'}")