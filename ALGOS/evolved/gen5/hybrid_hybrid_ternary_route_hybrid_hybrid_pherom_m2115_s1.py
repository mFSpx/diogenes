# DARWIN HAMMER — match 2115, survivor 1
# gen: 5
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s1.py (gen2)
# parent_b: hybrid_hybrid_pheromone_hyb_hybrid_hybrid_hybrid_m152_s0.py (gen4)
# born: 2026-05-29T23:40:50Z

"""
Hybrid Algorithm: Ternary Router + Minimum-Cost Tree + Pheromone Decay + SSIM Morphology

Parents:
- hybrid_ternary_router_hybrid_minimum_cost__m36_s1.py (tree cost & Bayesian update)
- hybrid_hybrid_pheromone_hyb_hybrid_hybrid_hybrid_m152_s0.py (morphology indices, pheromone decay, SSIM)

Mathematical Bridge:
Edge desirability is modeled as a Bayesian posterior where:
    prior   = decayed pheromone level (temporal relevance)
    likelihood = structural similarity between endpoint morphologies (SSIM)
    false_positive = uncertainty of the edge (inverse of recovery priority)

The posterior replaces the pure Bayesian marginal from Parent A and feeds directly into
the cost function of the ternary router.  Thus the hybrid cost of an edge becomes

    w_edge = length(a,b) + λ·posterior

where λ balances geometric distance and information‑theoretic confidence.
"""

import sys
import math
import random
import pathlib
from typing import Dict, Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Morphology and shape indices (from Parent B)
# ----------------------------------------------------------------------
class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(m: Morphology) -> float:
    """Sphericity = (V)^(1/3) / longest dimension."""
    V = m.length * m.width * m.height
    longest = max(m.length, m.width, m.height)
    return V ** (1.0 / 3.0) / longest

def flatness_index(m: Morphology) -> float:
    """Flatness = (length+width) / (2*height)."""
    return (m.length + m.width) / (2.0 * m.height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical model for righting time."""
    fi = flatness_index(m)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ----------------------------------------------------------------------
# Pheromone dynamics (from Parent B)
# ----------------------------------------------------------------------
def pheromone_decay(v0: float, half_life_seconds: int, delta_t: int) -> float:
    """Exponential decay based on half‑life."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    tau = half_life_seconds / 3600.0          # convert to hours
    return v0 * (0.5 ** (delta_t / tau))

# ----------------------------------------------------------------------
# Structural Similarity Index (SSIM) for morphology vectors (from Parent B)
# ----------------------------------------------------------------------
def ssim(x: List[float], y: List[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Simplified SSIM for 1‑D signals."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mu_x = x_arr.mean()
    mu_y = y_arr.mean()
    sigma_x = x_arr.var()
    sigma_y = y_arr.var()
    sigma_xy = ((x_arr - mu_x) * (y_arr - mu_y)).mean()
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)

# ----------------------------------------------------------------------
# Bayesian posterior that fuses pheromone (prior) and SSIM (likelihood)
# ----------------------------------------------------------------------
def bayesian_posterior(prior: float, likelihood: float,
                       false_positive: float) -> float:
    """
    Posterior = likelihood * prior / (likelihood * prior + false_positive * (1-prior))

    This is a normalized version of the classic Bayes rule where
    false_positive plays the role of the complement probability of the edge.
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0,1]")
    numerator = likelihood * prior
    denominator = numerator + false_positive * (1 - prior)
    if denominator == 0:
        return 0.0
    return numerator / denominator

# ----------------------------------------------------------------------
# Hybrid edge weight combining geometry, pheromone, and morphology similarity
# ----------------------------------------------------------------------
def hybrid_edge_weight(a: str, b: str,
                       nodes: Dict[str, Point],
                       morphologies: Dict[str, Morphology],
                       pheromones: Dict[Tuple[str, str], float],
                       half_life_seconds: int,
                       delta_t: int,
                       lambda_factor: float = 0.2) -> float:
    """
    Compute a unified weight for edge (a,b):
        w = Euclidean distance + λ * posterior

    posterior = Bayesian update where:
        prior      = decayed pheromone level (temporal relevance)
        likelihood = SSIM between morphology feature vectors of a and b
        false_pos  = 1 - recovery_priority (uncertainty of each node)
    """
    # geometric component
    geo = euclidean_length(nodes[a], nodes[b])

    # pheromone prior (decayed)
    raw_pheromone = pheromones.get((a, b), pheromones.get((b, a), 0.0))
    prior = pheromone_decay(raw_pheromone, half_life_seconds, delta_t)

    # morphology feature vectors (length, width, height, mass)
    vec_a = [morphologies[a].length, morphologies[a].width,
             morphologies[a].height, morphologies[a].mass]
    vec_b = [morphologies[b].length, morphologies[b].width,
             morphologies[b].height, morphologies[b].mass]

    likelihood = ssim(vec_a, vec_b)  # in [0,1]

    # false positive derived from recovery priority of both nodes
    fp_a = 1.0 - recovery_priority(morphologies[a])
    fp_b = 1.0 - recovery_priority(morphologies[b])
    false_positive = (fp_a + fp_b) / 2.0

    posterior = bayesian_posterior(prior, likelihood, false_positive)

    return geo + lambda_factor * posterior

# ----------------------------------------------------------------------
# Hybrid tree cost that aggregates hybrid edge weights
# ----------------------------------------------------------------------
def hybrid_tree_cost(nodes: Dict[str, Point],
                     edges: List[Edge],
                     root: str,
                     morphologies: Dict[str, Morphology],
                     pheromones: Dict[Tuple[str, str], float],
                     half_life_seconds: int,
                     delta_t: int,
                     path_weight: float = 0.2,
                     lambda_factor: float = 0.2) -> float:
    """
    Compute total cost of a spanning tree rooted at `root`.
    The cost consists of:
        - sum of hybrid edge weights (geometric + information term)
        - path_weight * sum of distances from root to every node (as in Parent A)
    """
    # Build adjacency for traversal
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    hybrid_edge_sum = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        hybrid_edge_sum += hybrid_edge_weight(
            a, b, nodes, morphologies, pheromones,
            half_life_seconds, delta_t, lambda_factor
        )

    # BFS/DFS to compute root distances
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                edge_len = euclidean_length(nodes[cur], nodes[nb])
                dist[nb] = dist[cur] + edge_len
                stack.append(nb)

    path_component = path_weight * sum(dist.values())
    return hybrid_edge_sum + path_component

# ----------------------------------------------------------------------
# Pheromone update based on node recovery priority (demonstrates interaction)
# ----------------------------------------------------------------------
def update_pheromones(pheromones: Dict[Tuple[str, str], float],
                     edges: List[Edge],
                     morphologies: Dict[str, Morphology],
                     reinforcement: float = 0.1) -> None:
    """
    For each edge, increase pheromone proportionally to the average
    recovery priority of its endpoints, then decay globally.
    The function mutates the `pheromones` dictionary in‑place.
    """
    # Reinforcement step
    for a, b in edges:
        rp_a = recovery_priority(morphologies[a])
        rp_b = recovery_priority(morphologies[b])
        inc = reinforcement * (rp_a + rp_b) / 2.0
        key = (a, b) if (a, b) in pheromones else (b, a)
        pheromones[key] = pheromones.get(key, 0.0) + inc

    # Global decay (using a fixed half‑life for simplicity)
    half_life_seconds = 3600  # 1 hour
    delta_t = 60              # 1 minute tick
    for key in list(pheromones.keys()):
        pheromones[key] = pheromone_decay(pheromones[key], half_life_seconds, delta_t)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph with three nodes
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0)
    }

    edges = [("A", "B"), ("A", "C"), ("B", "C")]

    morphologies = {
        "A": Morphology(1.0, 0.5, 0.3, 2.0),
        "B": Morphology(1.2, 0.4, 0.35, 2.1),
        "C": Morphology(0.9, 0.6, 0.25, 1.9)
    }

    # Initialise pheromones arbitrarily
    pheromones = {
        ("A", "B"): 0.8,
        ("A", "C"): 0.5,
        ("B", "C"): 0.3
    }

    # Run a single update‑cost cycle
    cost_before = hybrid_tree_cost(
        nodes, edges, "A",
        morphologies, pheromones,
        half_life_seconds=1800,   # 30 min half‑life
        delta_t=300                # 5 min elapsed
    )
    print(f"Hybrid tree cost before pheromone update: {cost_before:.4f}")

    # Update pheromones based on recovery priorities
    update_pheromones(pheromones, edges, morphologies)

    cost_after = hybrid_tree_cost(
        nodes, edges, "A",
        morphologies, pheromones,
        half_life_seconds=1800,
        delta_t=300
    )
    print(f"Hybrid tree cost after pheromone update:  {cost_after:.4f}")

    # Ensure no exception and reasonable output
    sys.exit(0)