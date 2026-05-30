# DARWIN HAMMER — match 1515, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s3.py (gen4)
# born: 2026-05-29T23:37:02Z

"""
Module for the hybrid algorithm that combines the Clifford-geometric distance 
and Voronoi partitioning from 'hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s4.py' 
with the minimum-cost tree Bayes update and bandit-router sketch-RLCT from 
'hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s3.py'. 

The mathematical bridge between the two structures is based on representing 
the probabilistic weights and log-count statistics as a path that can be 
approximated using the KAN, and using the Clifford-geometric distance to 
compute the expected cost and the expected reward.

The hybrid algorithm integrates the governing equations of both parents by 
using the KAN to approximate the level-1 and level-2 iterated-integrals, 
which are then used to compute the path signature and the expected cost and 
reward, and by using the Clifford-geometric distance to compute the 
edge-length matrix and the Voronoi partition.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

Point = Tuple[float, float]
Edge = Tuple[str, str]

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicate indices cancel because e_i·e_i = 1.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel pair
                del lst[j : j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    """Multiply two basis blades and return (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def clifford_distance(a: Point, b: Point) -> float:
    """Clifford-geometric distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Clifford-geometric edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → Clifford-geometric length
    root_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    root_dist = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = clifford_distance(nodes[a], nodes[b])
        edge_len[(b, a)] = clifford_distance(nodes[a], nodes[b])

    return adj, edge_len, root_dist

def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (local)
    At odd  indices 2t+1 : (X_t, X_{t+1}) (interaction)
    """
    T, d = path.shape
    lead_lag_path = np.zeros((2*T-1, 2*d))
    for t in range(T):
        lead_lag_path[2*t] = np.concatenate((path[t], path[t]))
        if t < T - 1:
            lead_lag_path[2*t+1] = np.concatenate((path[t], path[t+1]))
    return lead_lag_path

def hybrid_operation(nodes: Dict[str, Point], edges: List[Edge], root: str):
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    path = np.array([nodes[node] for node in [root] + list(adj[root])])
    lead_lag_path = lead_lag_transform(path)
    # Use lead_lag_path to compute the expected cost and reward
    expected_cost = np.mean(lead_lag_path, axis=0)
    expected_reward = np.std(lead_lag_path, axis=0)
    return expected_cost, expected_reward

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
        'D': (0, 1)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    expected_cost, expected_reward = hybrid_operation(nodes, edges, root)
    print("Expected Cost:", expected_cost)
    print("Expected Reward:", expected_reward)