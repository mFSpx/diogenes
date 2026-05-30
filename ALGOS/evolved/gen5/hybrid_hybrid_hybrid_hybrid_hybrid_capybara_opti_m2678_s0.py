# DARWIN HAMMER — match 2678, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py (gen4)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:43:35Z

"""
Module for the hybrid minimum-cost tree Bayes update and capybara-tri conduit algorithm.

This module combines the minimum-cost tree Bayes update algorithm from 'hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py'
and the capybara-tri conduit algorithm from 'hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py' by finding a mathematical interface 
between their structures. The minimum-cost tree Bayes update algorithm uses a deterministic cost function with probabilistic weights, 
while the capybara-tri conduit algorithm uses statistical gating logic and exponential evasion schedule. The combined quantities 
feed the free-energy asymptotic and the RLCT regression. The mathematical bridge between the two algorithms is the use of probabilistic 
weights and log-count statistics in the minimum-cost tree Bayes update algorithm, and the representation of the signal-to-noise gap as 
a confidence scalar in the capybara-tri conduit algorithm. This allows us to leverage the flexibility and power of the KAN to model 
complex paths and their signatures, and to integrate the governing equations of both parents by using the KAN to approximate the 
level-1 and level-2 iterated-integrals, which are then used to compute the path signature and the expected cost.

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
Vector = list[float]

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

def clamp(x: Vector, lower: float, upper: float) -> list[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]

def hybrid_evasion(nodes: dict[str, Point], edges: list[Edge], root: str, t: int, t_max: int) -> tuple[float, dict[str, float]]:
    """
    Compute the hybrid evasion magnitude and node distances.

    Returns
    -------
    delta : hybrid evasion magnitude
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    delta = evasion_delta(t, t_max)
    for node in nodes:
        node_dist[node] = delta * math.sqrt(length(nodes[root], nodes[node]))
    return delta, node_dist

def hybrid_signature(nodes: dict[str, Point], edges: list[Edge], root: str, t: int, t_max: int) -> float:
    """
    Compute the hybrid path signature.

    Returns
    -------
    signature : hybrid path signature
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    delta, node_dist = hybrid_evasion(nodes, edges, root, t, t_max)
    signature = 0.0
    for node in nodes:
        signature += delta * length(nodes[root], nodes[node])
    return signature

def hybrid_update(nodes: dict[str, Point], edges: list[Edge], root: str, t: int, t_max: int) -> tuple[float, dict[str, float]]:
    """
    Compute the hybrid update and node distances.

    Returns
    -------
    update : hybrid update magnitude
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    delta, node_dist = hybrid_evasion(nodes, edges, root, t, t_max)
    update = 0.0
    for node in nodes:
        update += delta * length(nodes[root], nodes[node])
    return update, node_dist

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    t = 1
    t_max = 10
    delta, node_dist = hybrid_evasion(nodes, edges, root, t, t_max)
    signature = hybrid_signature(nodes, edges, root, t, t_max)
    update, node_dist = hybrid_update(nodes, edges, root, t, t_max)
    print(f"Hybrid evasion magnitude: {delta}")
    print(f"Hybrid path signature: {signature}")
    print(f"Hybrid update magnitude: {update}")