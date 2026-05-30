# DARWIN HAMMER — match 1162, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s0.py (gen4)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s1.py (gen1)
# born: 2026-05-29T23:33:07Z

"""
This module introduces a novel HYBRID algorithm that mathematically fuses the governing equations of 
hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s0.py (Minimum-Cost Tree with Bayesian update) 
and hybrid_doomsday_calendar_gini_coefficient_m49_s1.py (Doomsday Calendar with Gini Coefficient). 
The connection is established by considering the Gini coefficient as a measure of inequality in the 
distribution of edge lengths in the Minimum-Cost Tree, and the Bayesian update as a means to inform 
the probabilistic transformation of the edge contributions. The hybrid algorithm enables the 
investigation of temporal patterns and inequality in edge length distributions.

The mathematical bridge is formed by using the Gini coefficient to weight the edge lengths in the 
Minimum-Cost Tree, and then applying the Bayesian update to the weighted edge lengths. This allows 
for the investigation of the impact of inequality in edge length distributions on the 
probabilistic transformation of the edge contributions.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
from datetime import date, timedelta

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

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
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist

def gini_coefficient(values: Iterable[float]) -> float:
    """Calculates the Gini coefficient of a given set of non-negative values."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: 
        return 0.0
    if xs[0] < 0: 
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

def bayesian_update(tree_dist: Dict[str, float], edge_len: Dict[Tuple[str, str], float], gini: float) -> Dict[str, float]:
    """
    Apply Bayesian update to the tree distances with Gini coefficient weighting.

    Parameters
    ----------
    tree_dist : dict mapping node → distance from *root*
    edge_len : dict mapping edge 
    gini : Gini coefficient of edge lengths
    """
    weighted_edge_len = {edge: gini * length for edge, length in edge_len.items()}
    updated_dist = {}
    for node, dist in tree_dist.items():
        updated_dist[node] = dist * (1 - gini) + sum(weighted_edge_len.get((node, neighbor), 0) for neighbor in tree_metrics({n: (0.0, 0.0) for n in tree_dist}, list(edge_len.keys()), list(tree_dist.keys())[0])[0][node])
    return updated_dist

def hybrid_doomsday_gini(tree_nodes: Dict[str, Tuple[float, float]], tree_edges: List[Tuple[str, str]], root: str, year: int, month: int, day: int) -> float:
    """This function calculates the Gini coefficient of the edge lengths in the Minimum-Cost Tree, 
    applies the Bayesian update to the tree distances, and then uses the Doomsday algorithm to 
    determine the weekday of the given date. The result is a weighted sum of the Gini coefficient, 
    the updated tree distances, and the weekday."""
    adj, edge_len, dist = tree_metrics(tree_nodes, tree_edges, root)
    gini = gini_coefficient(edge_len.values())
    updated_dist = bayesian_update(dist, edge_len, gini)
    doomsday = (date(year, month, day).weekday() + 1) % 7
    weight = gini / (1 + gini)
    return weight * gini + (1 - weight) * sum(updated_dist.values()) / len(updated_dist) + (1 - weight) * doomsday

def simulate_tree_inequality(year: int, month: int, day: int, num_days: int) -> float:
    """Simulates a tree inequality over a given period and calculates the corresponding 
    Gini coefficient."""
    tree_nodes = {f"node_{i}": (random.random(), random.random()) for i in range(10)}
    tree_edges = [(f"node_{i}", f"node_{(i+1)%10}") for i in range(10)]
    weekdays = []
    for i in range(num_days):
        date_day = date(year, month, day) + timedelta(days=i)
        weekdays.append((date_day.weekday() + 1) % 7)
    gini = gini_coefficient(weekdays)
    return hybrid_doomsday_gini(tree_nodes, tree_edges, "node_0", year, month, day)

if __name__ == "__main__":
    print(simulate_tree_inequality(2026, 5, 29, 10))