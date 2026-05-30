# DARWIN HAMMER — match 2678, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py (gen4)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:43:35Z

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Tuple, Dict, List
import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utilities 
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping ordered edge (a, b) → Euclidean length
    root_dist : dict mapping node → distance from `root`
    """
    adj = defaultdict(list)
    edge_len: Dict[Edge, float] = {}
    root_dist: Dict[str, float] = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        d = length(nodes[a], nodes[b])
        edge_len[(a, b)] = d
        edge_len[(b, a)] = d

    # BFS to compute root distances
    visited = {root}
    queue = [(root, 0.0)]
    while queue:
        cur, dist = queue.pop(0)
        root_dist[cur] = dist
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + edge_len[(cur, nb)]))

    return dict(adj), edge_len, root_dist

def edge_probabilities(
    nodes: Dict[str, Point],
    edges: List[Edge],
    beta: float = 1.0,
) -> Dict[Edge, float]:
    """
    Compute posterior probabilities `p_e ∝ exp(-β·c_e)` where `c_e` is the Euclidean
    length of edge `e`.  The probabilities are normalised over the undirected
    edge set (each unordered pair appears once in the input list).

    Returns
    -------
    dict mapping unordered edge (a, b) → probability.
    """
    if beta < 0:
        raise ValueError("beta must be non‑negative")
    costs = np.array([length(nodes[a], nodes[b]) for a, b in edges], dtype=float)
    unnorm = np.exp(-beta * costs)
    total = unnorm.sum()
    if total == 0:
        raise RuntimeError("All edge probabilities underflow to zero")
    probs = unnorm / total
    return {e: float(p) for e, p in zip(edges, probs)}

def tree_path_signature(
    nodes: Dict[str, Point],
    edges: List[Edge],
    level: int = 2,
) -> np.ndarray:
    """
    Compute a simple signature of the tree based on edge vectors.
    Level‑1: sum of edge vectors.
    Level‑2: sum of outer products of edge vectors (flattened).

    The signature is returned as a 1‑D NumPy array of length
    `2 + 4 = 6` for 2‑D points when `level >= 2`.  Higher levels are not
    implemented in this lightweight fusion.

    Parameters
    ----------
    nodes : dict mapping node id → (x, y)
    edges : list of unordered edges (a, b)
    level : maximal signature level (currently 1 or 2)

    Returns
    -------
    np.ndarray
    """
    if level < 1:
        raise ValueError("level must be >= 1")
    # Level‑1 accumulator
    v_sum = np.zeros(2, dtype=float)

    # Level‑2 accumulator (outer product flattened)
    outer_sum = np.zeros((2, 2), dtype=float)

    for a, b in edges:
        vec = np.array(nodes[b]) - np.array(nodes[a])
        v_sum += vec
        if level >= 2:
            outer_sum += np.outer(vec, vec)

    if level == 1:
        return v_sum
    else:
        return np.concatenate([v_sum, outer_sum.flatten()])

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def clamp(x: Vector, lower: float, upper: float) -> List[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]

def hoeffding_epsilon(n: int, delta: float = 0.05) -> float:
    """
    Hoeffding bound epsilon = sqrt( ln(2/δ) / (2n) ).
    """
    if n <= 0:
        return float('inf')
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))

def hybrid_optimization_step(
    nodes: Dict[str, Point],
    edges: List[Edge],
    t: int,
    t_max: int,
    observations: int,
    beta: float = 1.0,
    delta_max: float = 1.0,
    alpha: float = 3.0,
    confidence_scale: float = 1.0,
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
) -> Dict[str, Point]:
    """
    Perform a single hybrid optimisation iteration.

    1. Compute edge posterior probabilities and the normalised entropy Ĥ.
    2. Map Ĥ to a signal‑to‑noise gap Δ ∈ [0,1].
    3. Compute Hoeffding epsilon ε from the number of observations.
    4. Build the hybrid evasion magnitude
           δ_h = evasion_delta(t, t_max, delta_max, alpha) * (1 + ε) * Δ
    5. Perturb each node by a random vector scaled by δ_h and clamp.

    Returns a new dictionary with updated node coordinates
    """
    edge_probs = edge_probabilities(nodes, edges, beta)
    unnorm = np.array([edge_probs[e] for e in edges], dtype=float)
    total = unnorm.sum()
    probs = unnorm / total
    entropy = -np.sum(probs * np.log2(probs))
    normalised_entropy = entropy / np.log2(len(edges))
    delta = normalised_entropy
    epsilon = hoeffding_epsilon(observations)
    evasion = evasion_delta(t, t_max, delta_max, alpha)
    delta_h = evasion * (1 + epsilon) * delta
    new_nodes = {}
    for node, point in nodes.items():
        perturbation = np.random.uniform(-1, 1, size=2)
        new_point = np.array(point) + delta_h * perturbation
        new_point = clamp(new_point.tolist(), lower_bound, upper_bound)
        new_nodes[node] = tuple(new_point)
    return new_nodes

def improved_hybrid_optimization_step(
    nodes: Dict[str, Point],
    edges: List[Edge],
    t: int,
    t_max: int,
    observations: int,
    beta: float = 1.0,
    delta_max: float = 1.0,
    alpha: float = 3.0,
    confidence_scale: float = 1.0,
    lower_bound: float = -10.0,
    upper_bound: float = 10.0,
    tree_path_level: int = 2,
) -> Dict[str, Point]:
    """
    Perform a single hybrid optimisation iteration with improved integration.

    1. Compute edge posterior probabilities and the normalised entropy Ĥ.
    2. Map Ĥ to a signal‑to‑noise gap Δ ∈ [0,1].
    3. Compute Hoeffding epsilon ε from the number of observations.
    4. Build the hybrid evasion magnitude
           δ_h = evasion_delta(t, t_max, delta_max, alpha) * (1 + ε) * Δ
    5. Compute the tree path signature.
    6. Perturb each node by a random vector scaled by δ_h and the tree path signature, and clamp.

    Returns a new dictionary with updated node coordinates
    """
    edge_probs = edge_probabilities(nodes, edges, beta)
    unnorm = np.array([edge_probs[e] for e in edges], dtype=float)
    total = unnorm.sum()
    probs = unnorm / total
    entropy = -np.sum(probs * np.log2(probs))
    normalised_entropy = entropy / np.log2(len(edges))
    delta = normalised_entropy
    epsilon = hoeffding_epsilon(observations)
    evasion = evasion_delta(t, t_max, delta_max, alpha)
    delta_h = evasion * (1 + epsilon) * delta
    tree_signature = tree_path_signature(nodes, edges, tree_path_level)
    new_nodes = {}
    for node, point in nodes.items():
        perturbation = np.random.uniform(-1, 1, size=2)
        perturbation += tree_signature[:2]
        new_point = np.array(point) + delta_h * perturbation
        new_point = clamp(new_point.tolist(), lower_bound, upper_bound)
        new_nodes[node] = tuple(new_point)
    return new_nodes

# Example usage:
nodes = {
    'A': (0, 0),
    'B': (1, 0),
    'C': (1, 1),
    'D': (0, 1),
}
edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
t = 0
t_max = 10
observations = 100
beta = 1.0
delta_max = 1.0
alpha = 3.0
confidence_scale = 1.0
lower_bound = -10.0
upper_bound = 10.0
tree_path_level = 2

new_nodes = improved_hybrid_optimization_step(
    nodes, edges, t, t_max, observations, beta, delta_max, alpha, confidence_scale, lower_bound, upper_bound, tree_path_level
)
print(new_nodes)