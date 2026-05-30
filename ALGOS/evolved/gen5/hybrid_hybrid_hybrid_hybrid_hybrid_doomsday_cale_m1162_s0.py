# DARWIN HAMMER — match 1162, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s0.py (gen4)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s1.py (gen1)
# born: 2026-05-29T23:33:07Z

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
from datetime import date, timedelta

"""
This module integrates the hybrid_minimum_cost_tree_bayes_update_m6_s2 and 
doomsday_calendar_gini_coefficient algorithms into a single hybrid system. 
The mathematical bridge is formed by using the Gini coefficient as a measure 
of inequality in the distribution of node distances in the Minimum-Cost Tree, 
and the Doomsday algorithm as a means to determine the temporal patterns in the 
tree structure. The hybrid algorithm enables the investigation of temporal 
patterns and inequality in node distance distributions.

The core topology of the hybrid algorithm combines the tree metrics and Bayesian 
update from the hybrid_minimum_cost_tree_bayes_update_m6_s2 with the Gini 
coefficient calculation and doomsday algorithm from the doomsday_calendar_gini_coefficient.
"""

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

def gini_coefficient(values: List[float]) -> float:
    """Calculates the Gini coefficient of a given set of non-negative values."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: 
        return 0.0
    if xs[0] < 0: 
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

def doomsday(year: int, month: int, day: int) -> int:
    """Determines the weekday of a given date using the Doomsday algorithm."""
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    year -= month < 3
    return (year + int(year/4) - int(year/100) + int(year/400) + t[month-1] + day) % 7

def hybrid_tree_gini(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    year: int,
    month: int,
    day: int,
) -> float:
    """This function calculates the Gini coefficient of the node distances in the tree 
    and then applies the Doomsday algorithm to determine the weekday of the given date. 
    The result is a weighted sum of the Gini coefficient and the weekday."""
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    gini = gini_coefficient(list(dist.values()))
    doomsday_value = doomsday(year, month, day)
    weight = doomsday_value / 7
    return weight * gini + (1 - weight) * doomsday_value

def simulate_node_distribution(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    num_days: int,
) -> np.ndarray:
    """Simulates a node distance distribution over a given period and calculates the 
    corresponding Gini coefficient."""
    node_distances = []
    for i in range(num_days):
        year = 2024
        month = 1
        day = i % 31 + 1
        gini = hybrid_tree_gini(nodes, edges, root, year, month, day)
        node_distances.append(gini)
    return np.array(node_distances)

def calculate_temporal_inequality(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    num_days: int,
) -> float:
    """Calculates the temporal inequality in a node distance distribution over a 
    given period."""
    node_distances = simulate_node_distribution(nodes, edges, root, num_days)
    return gini_coefficient(node_distances)

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (3, 4),
        'C': (6, 8),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    year = 2024
    month = 1
    day = 1
    num_days = 30
    print(hybrid_tree_gini(nodes, edges, root, year, month, day))
    print(simulate_node_distribution(nodes, edges, root, num_days))
    print(calculate_temporal_inequality(nodes, edges, root, num_days))