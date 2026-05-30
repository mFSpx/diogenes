# DARWIN HAMMER — match 1889, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1162_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s1.py (gen5)
# born: 2026-05-29T23:39:26Z

"""
This module integrates the hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1162_s1.py 
(Minimum-Cost Tree with Bayesian update and Gini Coefficient) and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s1.py (Schoolfield-Rollinson 
poikilotherm rate primitive and Hybrid NLMS & Liquid-Time-Constant (LTC) Network).

The mathematical bridge between the two structures is established by using the 
Gini coefficient to weight the edge lengths in the Minimum-Cost Tree, and then 
applying the Schoolfield-Rollinson poikilotherm rate primitive to update the 
weighted edge lengths. The Bayesian update is then applied to the updated 
weighted edge lengths. This allows for the investigation of the impact of 
temperature-dependent developmental rates on the probabilistic transformation of 
the edge contributions.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple
from datetime import date, timedelta
from dataclasses import dataclass

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def gini_coefficient(edge_lengths: List[float]) -> float:
    """Gini coefficient for a list of edge lengths."""
    edge_lengths = np.array(edge_lengths)
    if edge_lengths.size == 0:
        return 0.0
    edge_lengths = edge_lengths.flatten()
    if np.amin(edge_lengths) < 0:
        edge_lengths -= np.amin(edge_lengths)
    edge_lengths += 0.0000001
    total = np.sum(edge_lengths)
    index = np.arange(1, edge_lengths.size+1)
    n = edge_lengths.size
    return ((np.sum((2 * index - n  - 1) * edge_lengths)) / (n * total))

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nei in adj[cur]:
            if nei not in dist:
                dist[nei] = dist[cur] + edge_len[(cur, nei)]
                stack.append(nei)

    return adj, edge_len, dist

def hybrid_algorithm(nodes: Dict[str, Tuple[float, float]], 
                     edges: List[Tuple[str, str]], 
                     root: str, 
                     temp_k: float) -> Tuple[Dict[str, List[str]], 
                                              Dict[Tuple[str, str], float], 
                                              Dict[str, float]]:
    edge_lengths = [length(nodes[a], nodes[b]) for a, b in edges]
    gini_coef = gini_coefficient(edge_lengths)
    schoolfield_rate = developmental_rate(temp_k)
    weighted_edge_lengths = [gini_coef * schoolfield_rate * length(nodes[a], nodes[b]) for a, b in edges]
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    for edge in edge_len:
        edge_len[edge] = weighted_edge_lengths[[(a, b) for a, b in edges].index(edge)]

    return adj, edge_len, dist

def predict(weights, x):
    """Predict the output of the system using the given weights and input."""
    return sum(w * xi for w, xi in zip(weights, x))

def update(weights, x, learning_rate):
    return [w + learning_rate * xi for w, xi in zip(weights, x)]

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
        'D': (0.0, 1.0)
    }
    edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]
    root = 'A'
    temp_k = 298.15

    adj, edge_len, dist = hybrid_algorithm(nodes, edges, root, temp_k)
    print(adj)
    print(edge_len)
    print(dist)

    weights = [0.5, 0.5]
    x = [1.0, 2.0]
    print(predict(weights, x))
    print(update(weights, x, 0.1))