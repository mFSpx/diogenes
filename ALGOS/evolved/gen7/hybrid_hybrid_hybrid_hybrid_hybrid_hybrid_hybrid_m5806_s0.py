# DARWIN HAMMER — match 5806, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s1.py (gen6)
# born: 2026-05-30T00:04:42Z

"""
This module integrates the core topologies of two mathematical algorithms: 
hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s0 and 
hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s1. 
The mathematical bridge between these two structures is found in their 
common goal of managing limited resources and predicting outcomes, 
specifically through the use of ternary tree loss and logistic regression 
from the former, and physarum flux and conductance dynamics from the latter. 
This module fuses these concepts by introducing a novel hybrid algorithm 
that integrates the governing equations of both parents, using the physarum 
flux to model the flow of resources through a network and the ternary tree 
loss to compute the expected reward of each action, and logistic regression 
to update the probabilistic weights.
"""

import numpy as np
import math
import random
import sys
import pathlib

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
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj = {}
    edge_len = {}
    node_dist = {}

    for a, b in edges:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])

    # Compute root-to-node distances using BFS
    queue = [(root, 0)]
    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node not in visited:
            visited.add(node)
            node_dist[node] = dist
            for neig in adj[node]:
                if neig not in visited:
                    queue.append((neig, dist + 1))

    return adj, edge_len, node_dist

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float,
                       dt: float = 0.1,
                       gain: float = 1.0,
                       decay: float = 0.01,
                       eps: float = 1e-12) -> float:
    return conductance + dt * (gain * q - decay * conductance)

def hybrid_update(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    conductance: float,
    pressure_a: float,
    pressure_b: float,
) -> tuple[dict[str, list[str]], dict[Edge, float], dict[str, float], float]:
    """
    Hybrid update function that combines tree metrics and physarum flux.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    conductance : updated conductance value
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    q = flux(conductance, edge_len[(root, random.choice(edges)[1])], pressure_a, pressure_b)
    conductance = update_conductance(conductance, q)
    return adj, edge_len, node_dist, conductance

def ternary_tree_loss(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
) -> float:
    """
    Ternary tree loss function.

    Returns
    -------
    loss : float value representing the loss
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    loss = 0
    for node in nodes:
        loss += node_dist[node]
    return loss

if __name__ == "__main__":
    nodes = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (0, 1),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    conductance = 1.0
    pressure_a = 1.0
    pressure_b = 0.0
    adj, edge_len, node_dist, conductance = hybrid_update(nodes, edges, root, conductance, pressure_a, pressure_b)
    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Node distances:", node_dist)
    print("Conductance:", conductance)
    print("Ternary tree loss:", ternary_tree_loss(nodes, edges, root))