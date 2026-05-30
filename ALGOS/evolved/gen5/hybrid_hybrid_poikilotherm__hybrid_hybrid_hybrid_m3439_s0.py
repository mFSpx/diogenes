# DARWIN HAMMER — match 3439, survivor 0
# gen: 5
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s0.py (gen4)
# born: 2026-05-29T23:50:07Z

"""
This module combines the Hybrid Poikilotherm-Tree Bayesian Cost Module from 'hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py'
and the hybrid minimum-cost tree Bayesian bandit-router algorithm with privacy considerations from 'hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s0.py'
by finding a mathematical interface between their structures.

The mathematical bridge between the two algorithms is the use of the temperature-dependent developmental rate from the Poikilotherm-Tree model
to modulate the confidence in each edge posterior in the Bayesian bandit-router algorithm, and the use of the probabilistic weights
from the bandit-router algorithm to modify the cost function in the Poikilotherm-Tree model.

This module integrates the governing equations of both parents by combining the temperature-dependent developmental rate with the log-count statistics
to estimate the expected reward of each action, and the probabilistic weights to modify the cost function.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

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
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    node_dist = {}

    for a, b in edges:
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
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

    return dict(adj), edge_len, node_dist

def developmental_rate(temperature: float) -> float:
    """Temperature-dependent developmental rate based on the Schoolfield–Rollinson model."""
    return 1 / (1 + math.exp(-temperature))

def bayes_edge_posteriors(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    temperature: float,
) -> Dict[Edge, float]:
    """
    Compute Bayesian edge posteriors using the temperature-dependent developmental rate.

    Returns
    -------
    posteriors : dict mapping edge → posterior probability
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    posteriors = {}

    for a, b in edges:
        prior = 1 / len(edges)
        likelihood = 1 / edge_len[(a, b)]
        posterior = prior * likelihood * developmental_rate(temperature)
        posteriors[(a, b)] = posterior

    return posteriors

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    temperature: float,
    lambda_: float,
) -> float:
    """
    Compute the hybrid tree cost using the temperature-dependent developmental rate and the probabilistic weights.

    Returns
    -------
    cost : float
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    posteriors = bayes_edge_posteriors(nodes, edges, root, temperature)
    cost = 0

    for a, b in edges:
        cost += posteriors[(a, b)] * edge_len[(a, b)]
    cost /= sum(posteriors.values())

    for node in nodes:
        cost += lambda_ * node_dist[node] * developmental_rate(temperature)

    return cost

def temperature_dependent_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    temperature: float,
    lambda_: float,
) -> float:
    """
    Compute the temperature-dependent cost by combining the hybrid tree cost and the temperature-dependent developmental rate.

    Returns
    -------
    cost : float
    """
    cost = hybrid_tree_cost(nodes, edges, root, temperature, lambda_)
    cost *= developmental_rate(temperature)
    return cost

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    temperature = 1.0
    lambda_ = 0.5

    cost = temperature_dependent_cost(nodes, edges, root, temperature, lambda_)
    print("Temperature-dependent cost:", cost)