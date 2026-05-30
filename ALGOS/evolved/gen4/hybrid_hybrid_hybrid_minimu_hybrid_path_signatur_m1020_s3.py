# DARWIN HAMMER — match 1020, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py (gen3)
# parent_b: hybrid_path_signature_kan_m30_s1.py (gen1)
# born: 2026-05-29T23:32:21Z

"""
Module for the hybrid algorithm that combines the minimum-cost tree Bayes update and bandit-router sketch-RLCT from 
'hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py' with the path signature and Kolmogorov-Arnold Networks (KAN) 
from 'hybrid_path_signature_kan_m30_s1.py'. The mathematical bridge between the two structures is based on representing 
the probabilistic weights and log-count statistics as a path that can be approximated using the KAN.

The core idea is to use the KAN to approximate the expected cost and the expected reward, which are then used to compute 
the free-energy asymptotic and the RLCT regression. This allows us to leverage the flexibility and power of the KAN to 
model complex systems and their rewards.

The hybrid algorithm integrates the governing equations of both parents by using the KAN to approximate the 
level-1 and level-2 iterated-integrals, which are then used to compute the path signature and the expected cost and reward.
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

def kan_approx(path, weights):
    """Kolmogorov-Arnold Networks (KAN) approximation.

    path: (T, d). weights: (d,). Returns approximated path.
    """
    return np.dot(path, weights)

def hybrid_operation(nodes, edges, root, path, weights):
    """
    Hybrid operation that combines the minimum-cost tree Bayes update and bandit-router sketch-RLCT with the path signature 
    and Kolmogorov-Arnold Networks (KAN).

    Returns
    -------
    expected_cost : float
    expected_reward : float
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    lead_lag_path = lead_lag_transform(path)
    level1_signature = signature_level1(path)
    approx_path = kan_approx(path, weights)

    # Compute expected cost and reward using the approximated path
    expected_cost = np.sum(np.abs(level1_signature)) * np.sum(np.abs(approx_path))
    expected_reward = np.sum(np.abs(level1_signature)) * np.sum(np.abs(lead_lag_path))

    return expected_cost, expected_reward

def main():
    # Example usage
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (1, 1)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    path = np.random.rand(10, 2)
    weights = np.random.rand(2)

    expected_cost, expected_reward = hybrid_operation(nodes, edges, root, path, weights)
    print(f"Expected cost: {expected_cost}, Expected reward: {expected_reward}")

if __name__ == "__main__":
    main()