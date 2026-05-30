# DARWIN HAMMER — match 2532, survivor 1
# gen: 5
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s2.py (gen4)
# born: 2026-05-29T23:42:44Z

"""Hybrid algorithm combining:
- Parent A: FairyFuse ternary router with Bayesian edge uncertainty (hybrid_tree_cost, bayes_update).
- Parent B: Caputo fractional derivative based minimum‑cost tree (caputo_derivative, gamma_lanczos).

Mathematical bridge:
The posterior probability of each edge (from the Bayesian update) is used as the order α
parameter in a Caputo fractional derivative that modulates the path‑weight term.
Thus edge uncertainty directly shapes the fractional attenuation of the accumulated
root‑to‑node distances, yielding a unified cost metric that respects both probabilistic
evidence and fractional dynamics.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple, Any

import numpy as np

Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Geometry utilities (from Parent A)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian primitives (from Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)"""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)"""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def edge_posterior(prior: float, likelihood: float, false_positive: float) -> float:
    """Convenience wrapper that returns the posterior probability for an edge."""
    m = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, m)

# ----------------------------------------------------------------------
# Lanczos Gamma and Caputo derivative (from Parent B)
# ----------------------------------------------------------------------
def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        z_minus_one = z - 1
        x = _LANCZOS_C[0]
        for i in range(1, len(_LANCZOS_C)):
            x += _LANCZOS_C[i] / (z_minus_one + i)
        t = z_minus_one + _LANCZOS_G + 0.5
        return math.sqrt(2 * math.pi) * t ** (z_minus_one + 0.5) * math.exp(-t) * x


def caputo_derivative(alpha: float, values: List[float]) -> float:
    """
    Discrete Caputo fractional derivative of order `alpha` for a sequence `values`.
    Implements the definition:
        D^α f(t) ≈ 1/Γ(1-α) * Σ_{τ=0}^{t-1} (f[t] - f[τ]) / (t-τ)^{α}
    where `t` is the last index (len(values)-1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1) for this discrete implementation")
    t = len(values) - 1
    if t < 1:
        return 0.0
    coeff = 1.0 / gamma_lanczos(1.0 - alpha)
    acc = 0.0
    ft = values[t]
    for tau in range(t):
        acc += (ft - values[tau]) / ((t - tau) ** alpha)
    return coeff * acc

# ----------------------------------------------------------------------
# Hybrid cost computation
# ----------------------------------------------------------------------
def fractional_path_weight(distances: List[float], alpha: float, base_weight: float = 0.2) -> float:
    """
    Compute a path‑weight factor that blends a constant base weight with a fractional
    contribution derived from the Caputo derivative of the distance sequence.
    """
    if not distances:
        return base_weight
    derivative = caputo_derivative(alpha, distances)
    # Normalise derivative to a small factor to keep the weight sensible
    norm_factor = 1.0 / (1.0 + abs(derivative))
    return base_weight * norm_factor


def hybrid_fractional_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    edge_likelihoods: Dict[Edge, float],
    false_positive: float = 0.05,
    alpha: float = 0.5,
    base_path_weight: float = 0.2,
) -> float:
    """
    Unified tree cost:
    - Material cost = Σ length(e) * posterior(e)
    - Path cost   = fractional_path_weight(distances, α) * Σ distances
    Edge posterior is obtained via Bayesian update using the supplied priors
    and likelihoods.
    """
    # Build adjacency and compute posterior‑weighted material cost
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        prior = edge_priors.get((a, b), edge_priors.get((b, a), 0.5))
        likelihood = edge_likelihoods.get((a, b), edge_likelihoods.get((b, a), 0.5))
        post = edge_posterior(prior, likelihood, false_positive)
        material += length(nodes[a], nodes[b]) * post

    # Breadth‑first traversal to obtain root‑to‑node distances
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + length(nodes[cur], nodes[nb])
                stack.append(nb)

    distance_values = list(dist.values())
    path_weight = fractional_path_weight(distance_values, alpha, base_path_weight)

    return material + path_weight * sum(distance_values)


# ----------------------------------------------------------------------
# Supporting utilities (demonstrating hybrid operation)
# ----------------------------------------------------------------------
def generate_random_graph(num_nodes: int, connectivity: int = 2) -> Tuple[Dict[str, Point], List[Edge], str]:
    """
    Produce a random geometric graph:
    - Nodes are placed uniformly in the unit square.
    - Each node connects to its `connectivity` nearest neighbours.
    Returns nodes dict, edge list, and a randomly chosen root.
    """
    if num_nodes < 2:
        raise ValueError("need at least two nodes")
    nodes = {f"N{i}": (random.random(), random.random()) for i in range(num_nodes)}
    edges: List[Edge] = []
    for i, (nid, pt) in enumerate(nodes.items()):
        # compute distances to all other nodes
        dists = [(length(pt, nodes[other]), other) for other in nodes if other != nid]
        dists.sort(key=lambda x: x[0])
        for _, other in dists[:connectivity]:
            edge = tuple(sorted((nid, other)))
            if edge not in edges:
                edges.append(edge)
    root = random.choice(list(nodes.keys()))
    return nodes, edges, root


def random_edge_priors(edges: List[Edge]) -> Dict[Edge, float]:
    """Assign a random prior probability in (0,1) to each edge."""
    return {edge: random.uniform(0.3, 0.9) for edge in edges}


def random_edge_likelihoods(edges: List[Edge]) -> Dict[Edge, float]:
    """Assign a random likelihood in (0,1) to each edge."""
    return {edge: random.uniform(0.4, 0.95) for edge in edges}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a modest random graph
    nodes, edges, root = generate_random_graph(num_nodes=8, connectivity=3)

    # Random Bayesian parameters
    priors = random_edge_priors(edges)
    likelihoods = random_edge_likelihoods(edges)

    # Compute hybrid cost
    cost = hybrid_fractional_tree_cost(
        nodes=nodes,
        edges=edges,
        root=root,
        edge_priors=priors,
        edge_likelihoods=likelihoods,
        false_positive=0.05,
        alpha=0.6,
        base_path_weight=0.2,
    )

    print(f"Root node: {root}")
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")
    print(f"Hybrid fractional tree cost: {cost:.4f}")