# DARWIN HAMMER — match 2678, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py (gen4)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:43:35Z

"""Hybrid Minimum‑Cost Tree & Capybara‑Tri Conduit Algorithm.

Parents
-------
- **Parent A**: `hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py`
  Provides tree metrics, probabilistic edge weights from a minimum‑cost Bayes update,
  and a path‑signature representation (iterated integrals of edge vectors).

- **Parent B**: `hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py`
  Supplies a continuous optimisation schedule (`evasion_delta`), a Hoeffding‑based
  confidence (`ε`), and vector clamping utilities.

Mathematical Bridge
-------------------
The bridge is the *confidence scalar* derived from the probabilistic edge
distribution of the minimum‑cost tree.  Let  


p_e = exp(-β·c_e) / Σ_{e'} exp(-β·c_{e'})


be the posterior probability of edge `e` with cost `c_e` (Euclidean length).
The Shannon entropy  


H = - Σ_e p_e log p_e


measures uncertainty of the tree structure.  We map `H` to a *signal‑to‑noise*
gap `Δ` that rescales the evasion magnitude from Parent B:


Δ = H / log|E|          # normalised entropy ∈ [0,1]
δ_h(t) = evasion_delta(t, t_max) * (1 + ε) * Δ


Thus the optimisation step that moves node positions is directly driven by the
tree‑based confidence, while the Hoeffding epsilon `ε` (computed from the number
of observations) provides the statistical gating.  The path signature vector
computed from the tree supplies a feature that can be used as the “signal”
in downstream learning tasks.

The module implements three core hybrid operations:
1. `edge_probabilities` – Bayes‑style posterior over edges.
2. `tree_path_signature` – level‑1 and level‑2 iterated‑integral signature.
3. `hybrid_optimization_step` – moves node coordinates using the hybrid
   evasion magnitude `δ_h(t)` and clamps them to a feasible box.
"""

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
# Utilities from Parent A (tree metrics)
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


# ----------------------------------------------------------------------
# Hybrid component 1 – probabilistic edge weights (minimum‑cost Bayes)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Hybrid component 2 – path signature (level‑1 & level‑2 iterated integrals)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Hybrid component 3 – optimisation step driven by tree confidence
# ----------------------------------------------------------------------
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

    Returns a new dictionary with updated node coordinates.
    """
    # 1. Edge probabilities
    probs = edge_probabilities(nodes, edges, beta=beta)

    # 2. Normalised entropy
    prob_vals = np.array(list(probs.values()))
    # protect against log(0)
    prob_vals = np.clip(prob_vals, 1e-15, 1.0)
    entropy = -np.sum(prob_vals * np.log(prob_vals))
    max_entropy = math.log(len(prob_vals)) if len(prob_vals) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0  # ∈[0,1]

    delta_gap = normalized_entropy * confidence_scale  # Δ

    # 3. Hoeffding epsilon
    eps = hoeffding_epsilon(observations)

    # 4. Hybrid evasion magnitude
    base_delta = evasion_delta(t, t_max, delta_max, alpha)
    delta_h = base_delta * (1.0 + eps) * delta_gap

    # 5. Random perturbation
    new_nodes: Dict[str, Point] = {}
    for nid, (x, y) in nodes.items():
        # isotropic random direction
        angle = random.random() * 2.0 * math.pi
        dx = delta_h * math.cos(angle)
        dy = delta_h * math.sin(angle)
        nx, ny = clamp([x + dx, y + dy], lower_bound, upper_bound)
        new_nodes[nid] = (nx, ny)

    return new_nodes


# ----------------------------------------------------------------------
# Additional helper – gain gap (optional, demonstrates deeper fusion)
# ----------------------------------------------------------------------
def gain_gap(signal: float, noise: float) -> float:
    """
    Compute the signal‑to‑noise gap Δ = signal - noise.
    Used as an alternative scaling factor for the optimisation step.
    """
    return max(0.0, signal - noise)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree: a triangle
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, math.sqrt(3) / 2),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    # Compute metrics
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Root distances:", root_dist)

    # Edge probabilities
    probs = edge_probabilities(nodes, edges, beta=2.0)
    print("Edge probabilities:", probs)

    # Path signature
    sig = tree_path_signature(nodes, edges, level=2)
    print("Path signature (level 2):", sig)

    # Perform a hybrid optimisation step
    updated_nodes = hybrid_optimization_step(
        nodes,
        edges,
        t=5,
        t_max=20,
        observations=100,
        beta=2.0,
        delta_max=0.5,
        alpha=2.5,
        confidence_scale=1.0,
        lower_bound=-5.0,
        upper_bound=5.0,
    )
    print("Updated node positions:", updated_nodes)

    # Verify that the signature after the move is still computable
    new_sig = tree_path_signature(updated_nodes, edges, level=2)
    print("New path signature:", new_sig)