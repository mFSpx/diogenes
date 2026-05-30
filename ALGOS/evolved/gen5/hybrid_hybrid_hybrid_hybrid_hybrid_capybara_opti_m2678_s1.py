# DARWIN HAMMER — match 2678, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py (gen4)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:43:35Z

"""
Module for the hybrid Capybara-Tri Conduit Bayes update and path signature algorithm.

This module combines the Capybara-Tri Conduit algorithm from 'hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py'
with the hybrid minimum-cost tree Bayes update and path signature algorithm from 'hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py'
by finding a mathematical interface between their structures.
The Capybara-Tri Conduit algorithm uses a confidence scalar and a hybrid evasion magnitude,
while the hybrid minimum-cost tree Bayes update and path signature algorithm uses probabilistic weights and log-count statistics.
The mathematical bridge between the two algorithms is the use of the confidence scalar as a probabilistic weight and
the hybrid evasion magnitude as a parameter in the level-1 and level-2 iterated-integrals.
This allows us to leverage the flexibility and power of the KAN to model complex paths and their signatures,
and to integrate the governing equations of both parents by using the KAN to approximate the level-1 and level-2 iterated-integrals.
"""

import math
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import random

# Types
Point = tuple[float, float]
Edge = tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
) -> tuple[dict[str, list[str]], dict[Edge, float], dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → Euclidean length
    root_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    root_dist = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])

    return adj, edge_len, root_dist

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def clamp(x: list[float], lower: float, upper: float) -> list[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]

def hybrid_evasion_magnitude(signal: float, noise: float, epsilon: float, t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """
    Compute the hybrid evasion magnitude.

    Parameters
    ----------
    signal : float
        Signal value
    noise : float
        Noise value
    epsilon : float
        Hoeffding epsilon
    t : int
        Current time step
    t_max : int
        Maximum time step
    delta_max : float, optional
        Maximum evasion magnitude, by default 1.0
    alpha : float, optional
        Exponential decay rate, by default 3.0

    Returns
    -------
    float
        Hybrid evasion magnitude
    """
    confidence_scalar = signal - noise
    hybrid_evasion_magnitude = evasion_delta(t, t_max, delta_max, alpha) * (1 + epsilon)
    return hybrid_evasion_magnitude

def path_signature(adj: dict[str, list[str]], edge_len: dict[Edge, float], root: str) -> list[float]:
    """
    Compute the path signature.

    Parameters
    ----------
    adj : dict[str, list[str]]
        Adjacency list
    edge_len : dict[Edge, float]
        Edge lengths
    root : str
        Root node

    Returns
    -------
    list[float]
        Path signature
    """
    path_signature = []
    for node in adj:
        if node != root:
            path_signature.append(edge_len[(root, node)])
    return path_signature

def hybrid_cost_function(adj: dict[str, list[str]], edge_len: dict[Edge, float], root: str, path_signature: list[float]) -> float:
    """
    Compute the hybrid cost function.

    Parameters
    ----------
    adj : dict[str, list[str]]
        Adjacency list
    edge_len : dict[Edge, float]
        Edge lengths
    root : str
        Root node
    path_signature : list[float]
        Path signature

    Returns
    -------
    float
        Hybrid cost function
    """
    hybrid_evasion_magnitude = hybrid_evasion_magnitude(10, 2, 0.5, 10, 100)
    cost_function = 0
    for node in adj:
        if node != root:
            cost_function += edge_len[(root, node)] + hybrid_evasion_magnitude * path_signature[adj[node].index(node)]
    return cost_function

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    adj, edge_len, root_dist = tree_metrics(nodes, edges, "A")
    path_signature_values = path_signature(adj, edge_len, "A")
    hybrid_cost = hybrid_cost_function(adj, edge_len, "A", path_signature_values)
    print(hybrid_cost)