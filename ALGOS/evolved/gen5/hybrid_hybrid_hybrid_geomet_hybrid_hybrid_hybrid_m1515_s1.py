# DARWIN HAMMER — match 1515, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s3.py (gen4)
# born: 2026-05-29T23:37:02Z

import numpy as np
import math
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

Point = Tuple[float, float]
Edge = Tuple[str, str]

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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
                del lst[j : j + 2]
                n -= 2
                i = -1  
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def clifford_distance(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    adj = defaultdict(list)
    edge_len = {}
    root_dist = {root: 0}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = clifford_distance(nodes[a], nodes[b])
        edge_len[(b, a)] = clifford_distance(nodes[a], nodes[b])

    for node in adj:
        if node != root:
            root_dist[node] = min(edge_len[(node, neighbor)] + root_dist.get(neighbor, float('inf')) for neighbor in adj[node] if neighbor in root_dist)

    return adj, edge_len, root_dist

def lead_lag_transform(path):
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
    expected_cost = np.mean(np.linalg.norm(lead_lag_path, axis=1))
    expected_reward = np.std(np.linalg.norm(lead_lag_path, axis=1))
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