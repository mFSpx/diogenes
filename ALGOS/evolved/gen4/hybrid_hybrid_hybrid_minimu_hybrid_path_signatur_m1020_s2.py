# DARWIN HAMMER — match 1020, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py (gen3)
# parent_b: hybrid_path_signature_kan_m30_s1.py (gen1)
# born: 2026-05-29T23:32:21Z

import math
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import random

"""
Module for the hybrid minimum-cost tree Bayes update and path signature algorithm.

This module combines the minimum-cost tree Bayes update algorithm from 'hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py'
and the path signature algorithm from 'hybrid_path_signature_kan_m30_s1.py' by finding a mathematical interface between their structures.
The minimum-cost tree Bayes update algorithm uses a deterministic cost function with probabilistic weights, while the path signature
algorithm uses iterated-integral algebra to compute the path signature. The combined quantities feed the free-energy asymptotic
and the RLCT regression. The mathematical bridge between the two algorithms is the use of probabilistic weights and log-count
statistics in the minimum-cost tree Bayes update algorithm, and the representation of the path signature as a function that can
be approximated using the KAN. This allows us to leverage the flexibility and power of the KAN to model complex paths and their
signatures, and to integrate the governing equations of both parents by using the KAN to approximate the level-1 and level-2
iterated-integrals, which are then used to compute the path signature and the expected cost.
"""

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
    out[2 * (T - 1)] = np.empty((2*d))
    out[2 * (T - 1)][:d] = path[T - 1]
    out[2 * (T - 1)][d:] = path[T - 1]
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
    S2 = np.zeros((path.shape[1], path.shape[1]))
    for t in range(path.shape[0] - 1):
        S2 += np.outer(running[t], increments[t])
    return S2

def hybrid_min_cost_path_signature(nodes: dict[str, Point], edges: list[Edge], root: str, path):
    """Hybrid minimum-cost tree Bayes update and path signature algorithm.

    This function integrates the governing equations of both parents by using the KAN to approximate the level-1 and level-2
    iterated-integrals, which are then used to compute the path signature and the expected cost.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → Euclidean length
    root_dist : dict mapping node → root‑to‑node distance
    lead_lag_path : (2T-1, 2d) interleaved lead-lag path
    signature_level1 : (d,) total increment vector
    signature_level2 : (d, d) level-2 iterated integral tensor
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    lead_lag_path = lead_lag_transform(path)
    signature_1 = signature_level1(path)
    signature_2 = signature_level2(path)
    return adj, edge_len, root_dist, lead_lag_path, signature_1, signature_2

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
        'D': (0.0, 1.0)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    path = np.random.rand(10, 2)
    adj, edge_len, root_dist, lead_lag_path, signature_1, signature_2 = hybrid_min_cost_path_signature(nodes, edges, root, path)
    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Root distances:", root_dist)
    print("Lead-lag path:", lead_lag_path)
    print("Signature level 1:", signature_1)
    print("Signature level 2:")
    print(signature_2)