# DARWIN HAMMER — match 3587, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2366_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1161_s2.py (gen4)
# born: 2026-05-29T23:50:47Z

"""Hybrid Fisher-Pheromone Graph Module

This module fuses the mathematical cores of two parent algorithms:

* **Parent A** – computes a pheromone probability distribution, evaluates Fisher
  information on those probabilities and returns the entropy of the product
  `p * Fisher(p)` (see `hybrid_fisher_pheromone_entropy`).

* **Parent B** – builds a geometric tree, evaluates Fisher information on edge
  distances and incorporates it as a multiplicative curvature factor in the
  tree‑cost (see `tree_cost`).

**Mathematical Bridge**

Both parents share the *Fisher score*:


I(θ) = (∂/∂θ log f(θ))² / f(θ)


where `f(θ)` is a Gaussian‑like intensity (`gaussian_beam`).  
Parent A multiplies a probability `p` by `I(p)` before entropy, while
Parent B multiplies a distance `d` by `1 + I(d)` before summation.

The hybrid algorithm applies the Fisher score **simultaneously** to
probabilistic (pheromone) and geometric (distance) quantities on each graph
edge, producing a unified edge metric:


w_edge = d * (1 + I(d; μ_d, σ_d)) * (p * I(p; μ_p, σ_p))


The total cost is the sum of `w_edge` over all edges, and an overall entropy
is computed from the normalized edge weights, preserving both parents’
 information‑theoretic and curvature‑aware aspects.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared mathematical primitives
# ----------------------------------------------------------------------
def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher score of a Gaussian‑like intensity at `theta`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    intensity = max(math.exp(-0.5 * z * z), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability mass function."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def calculate_pheromone_probabilities(surface_key: str, limit: int) -> List[float]:
    """Generate a random normalized pheromone distribution."""
    random.seed(surface_key)  # deterministic per surface_key
    pheromones = [random.random() for _ in range(limit)]
    total = sum(pheromones)
    return [p / total for p in pheromones]


def euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Hybrid edge metric
# ----------------------------------------------------------------------
def hybrid_edge_metric(
    distance: float,
    pheromone_prob: float,
    fisher_center_dist: float,
    fisher_width_dist: float,
    fisher_center_prob: float,
    fisher_width_prob: float,
) -> float:
    """
    Compute the combined edge weight.

    w = d * (1 + I_d) * (p * I_p)

    where
        d  – geometric distance,
        p  – pheromone probability,
        I_d = fisher_score(d; fisher_center_dist, fisher_width_dist),
        I_p = fisher_score(p; fisher_center_prob, fisher_width_prob).
    """
    I_d = fisher_score(distance, fisher_center_dist, fisher_width_dist)
    I_p = fisher_score(pheromone_prob, fisher_center_prob, fisher_width_prob)
    return distance * (1.0 + I_d) * (pheromone_prob * I_p)


# ----------------------------------------------------------------------
# Hybrid tree cost using both pheromone and distance Fisher information
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    surface_key: str,
    limit: int,
    fisher_center_dist: float = 0.0,
    fisher_width_dist: float = 1.0,
    fisher_center_prob: float = 0.0,
    fisher_width_prob: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute a cost for a spanning tree that blends distance‑based Fisher curvature
    with pheromone‑based Fisher weighting.

    Returns:
        total_cost – sum of hybrid edge metrics,
        entropy_cost – Shannon entropy of the normalized edge metrics.
    """
    # 1️⃣ Pheromone probabilities per edge (deterministic via surface_key)
    pheromone_probs = calculate_pheromone_probabilities(surface_key, len(edges))

    # 2️⃣ Build adjacency for traversal
    adjacency: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    # 3️⃣ Depth‑first traversal to ensure a tree (ignore cycles)
    visited = set()
    stack = [root]
    edge_weights: List[float] = []

    while stack:
        current = stack.pop()
        visited.add(current)
        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            # Edge identifier (order‑independent) to fetch pheromone prob
            edge_idx = edges.index((current, neighbor)) if (current, neighbor) in edges else edges.index((neighbor, current))
            prob = pheromone_probs[edge_idx]

            dist = euclidean_length(nodes[current], nodes[neighbor])
            w = hybrid_edge_metric(
                distance=dist,
                pheromone_prob=prob,
                fisher_center_dist=fisher_center_dist,
                fisher_width_dist=fisher_width_dist,
                fisher_center_prob=fisher_center_prob,
                fisher_width_prob=fisher_width_prob,
            )
            edge_weights.append(w)
            stack.append(neighbor)

    total_cost = sum(edge_weights)
    entropy_cost = entropy(edge_weights)
    return total_cost, entropy_cost


# ----------------------------------------------------------------------
# Stand‑alone entropy of pheromone‑Fisher product (Parent A analogue)
# ----------------------------------------------------------------------
def hybrid_fisher_pheromone_entropy(
    surface_key: str,
    limit: int,
    center: float,
    width: float,
) -> float:
    """
    Compute entropy of the product p * I(p) where p are pheromone probabilities.
    Mirrors Parent A's `hybrid_fisher_pheromone`.
    """
    probs = calculate_pheromone_probabilities(surface_key, limit)
    fisher_vals = [fisher_score(p, center, width) for p in probs]
    combined = [p * f for p, f in zip(probs, fisher_vals)]
    return entropy(combined)


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple triangle graph
    nodes_example = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, math.sqrt(3) / 2),
    }
    edges_example = [("A", "B"), ("B", "C"), ("C", "A")]

    # Parameters
    surface = "test_surface"
    limit = len(edges_example)
    fisher_center_dist = 0.0
    fisher_width_dist = 1.0
    fisher_center_prob = 0.0
    fisher_width_prob = 1.0

    # Hybrid tree cost (root at A)
    cost, ent = hybrid_tree_cost(
        nodes=nodes_example,
        edges=edges_example,
        root="A",
        surface_key=surface,
        limit=limit,
        fisher_center_dist=fisher_center_dist,
        fisher_width_dist=fisher_width_dist,
        fisher_center_prob=fisher_center_prob,
        fisher_width_prob=fisher_width_prob,
    )
    print(f"Hybrid tree total cost: {cost:.6f}")
    print(f"Hybrid tree entropy: {ent:.6f}")

    # Entropy of pheromone‑Fisher product (Parent A component)
    ent_pheromone = hybrid_fisher_pheromone_entropy(
        surface_key=surface,
        limit=limit,
        center=0.0,
        width=1.0,
    )
    print(f"Pheromone‑Fisher entropy: {ent_pheromone:.6f}")

    # Verify that functions run without raising
    assert cost > 0 and ent > 0 and ent_pheromone > 0, "Metrics should be positive"
    print("Smoke test passed.")