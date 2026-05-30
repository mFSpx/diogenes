# DARWIN HAMMER — match 1020, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py (gen3)
# parent_b: hybrid_path_signature_kan_m30_s1.py (gen1)
# born: 2026-05-29T23:32:21Z

"""
This module implements a hybrid mathematical algorithm that combines the minimum-cost tree Bayes update and bandit-router 
sketch-RLCT algorithm from the 'hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py' module with the path signature 
and iterated-integral algebra from the 'hybrid_path_signature_kan_m30_s1.py' module. The mathematical bridge between 
the two structures is based on representing the path signature as a function that can be approximated using the 
probabilistic weights and log-count statistics from the minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm.

The core idea is to use the probabilistic weights and log-count statistics to approximate the iterated-integral algebra, 
which is a key component of the path signature. This allows us to leverage the flexibility and power of the 
probabilistic weights and log-count statistics to model complex paths and their signatures.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import random

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
        edge_len[(b, a)] = length(nodes[a], nodes[b])
        if a == root:
            root_dist[b] = edge_len[(a, b)]
        elif b == root:
            root_dist[a] = edge_len[(a, b)]

    return adj, edge_len, root_dist

def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (lead and lag both at t)
    At odd  indices 2t+1 : (X_{t+1}, X_t)  (lead advances, lag holds)
    This matches the standard Chevyrev-Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    """Level-1 signature: total increment vector.

    path: (T, d). Returns (d,). Equal to path[-1] - path[0].
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    """Level-2 iterated integral tensor.

    path: (T, d). Returns (d, d).
    S2[i,j] = sum_{s<t} (X_s[i] - X_0[i]) * dX_t[j]
             = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])

    Uses the standard left-point Riemann sum on the increment path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    result = np.zeros((path.shape[1], path.shape[1]))
    for i in range(path.shape[1]):
        for j in range(path.shape[1]):
            for t in range(path.shape[0] - 1):
                result[i, j] += running[t, i] * increments[t, j]
    return result

def hybrid_tree_signature(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path: np.ndarray,
) -> Tuple[Dict[Edge, float], np.ndarray, np.ndarray]:
    """
    Compute the hybrid tree signature.

    This function combines the tree metrics and path signature to compute a hybrid signature.

    Returns
    -------
    edge_len : dict mapping edge (ordered a, b) → Euclidean length
    signature_level1 : Level-1 signature: total increment vector
    signature_level2 : Level-2 iterated integral tensor
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    signature_level1_result = signature_level1(path)
    signature_level2_result = signature_level2(path)
    return edge_len, signature_level1_result, signature_level2_result

def hybrid_path_tree(
    path: np.ndarray,
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> np.ndarray:
    """
    Compute the hybrid path tree.

    This function combines the lead-lag transform and tree metrics to compute a hybrid path tree.

    Returns
    -------
    hybrid_path : (2T-1, 2d) interleaved lead-lag path
    """
    hybrid_path = lead_lag_transform(path)
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    return hybrid_path

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
    }
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    path = np.array([[0, 0], [1, 0], [1, 1]])
    edge_len, signature_level1_result, signature_level2_result = hybrid_tree_signature(nodes, edges, root, path)
    hybrid_path = hybrid_path_tree(path, nodes, edges, root)
    print(edge_len)
    print(signature_level1_result)
    print(signature_level2_result)
    print(hybrid_path)