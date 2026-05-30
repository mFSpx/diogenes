# DARWIN HAMMER — match 1020, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py (gen3)
# parent_b: hybrid_path_signature_kan_m30_s1.py (gen1)
# born: 2026-05-29T23:32:21Z

"""
Module for the hybrid minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm, combined with the path signature and iterated-integral algebra from the 'path_signature.py' and 'kan.py' modules.

This module integrates the mathematical structures of both parents by using the probabilistic weights from the minimum-cost tree Bayes update algorithm to approximate the expected reward of each action from the path signature.

The mathematical bridge between the two structures is based on representing the path signature as a function that can be approximated using the probabilistic weights. The minimum-cost tree Bayes update algorithm uses probabilistic weights to compute the expected cost, while the path signature uses log-count statistics to estimate the expected reward. By combining these two approaches, we can create a hybrid algorithm that uses probabilistic weights and log-count statistics to compute the expected cost and the expected reward.
"""

import math
import sys
import random
import pathlib
import numpy as np
from collections import defaultdict

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

# Algorithm A – deterministic tree utilities
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
        edge_len[(b, a)] = length(nodes[b], nodes[a])
        root_dist[a] = length(nodes[root], nodes[a])
        root_dist[b] = length(nodes[root], nodes[b])

    return adj, edge_len, root_dist

# Algorithm B – path signature and iterated-integral algebra
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
    # S2[i,j] = sum_t 

def hybrid_signature(path, nodes, edges, root):
    """
    Compute the level-2 iterated integral tensor using the probabilistic weights from the minimum-cost tree Bayes update algorithm.

    Parameters
    ----------
    path : (T, d) numpy array
        The input path.
    nodes : Dict[str, Point]
        The node positions.
    edges : List[Edge]
        The edges of the tree.
    root : str
        The root node.

    Returns
    -------
    S2 : (d, d) numpy array
        The level-2 iterated integral tensor.
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    probabilities = np.zeros((len(nodes),))
    for node in nodes:
        probabilities[node] = 1 / len(nodes)
    S2 = np.zeros((len(nodes), len(nodes)))
    for t in range(len(path) - 1):
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if adj[i].__contains__(j):
                    S2[i, j] += (path[t, i] - path[0, i]) * (path[t + 1, j] - path[t, j])
    return S2

def hybrid_path_signature(path, nodes, edges, root):
    """
    Compute the path signature using the probabilistic weights from the minimum-cost tree Bayes update algorithm.

    Parameters
    ----------
    path : (T, d) numpy array
        The input path.
    nodes : Dict[str, Point]
        The node positions.
    edges : List[Edge]
        The edges of the tree.
    root : str
        The root node.

    Returns
    -------
    signature : (d,) numpy array
        The path signature.
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    probabilities = np.zeros((len(nodes),))
    for node in nodes:
        probabilities[node] = 1 / len(nodes)
    signature = np.zeros((len(nodes),))
    for i in range(len(nodes)):
        signature[i] = path[-1, i] - path[0, i]
    return signature

def hybrid_level2(path, nodes, edges, root):
    """
    Compute the level-2 iterated integral tensor using the lead-lag transform and the probabilistic weights from the minimum-cost tree Bayes update algorithm.

    Parameters
    ----------
    path : (T, d) numpy array
        The input path.
    nodes : Dict[str, Point]
        The node positions.
    edges : List[Edge]
        The edges of the tree.
    root : str
        The root node.

    Returns
    -------
    S2 : (d, d) numpy array
        The level-2 iterated integral tensor.
    """
    lead_lag_path = lead_lag_transform(path)
    S2 = np.zeros((len(nodes), len(nodes)))
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    probabilities = np.zeros((len(nodes),))
    for node in nodes:
        probabilities[node] = 1 / len(nodes)
    for t in range(len(lead_lag_path) - 1):
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if adj[i].__contains__(j):
                    S2[i, j] += (lead_lag_path[t, 2 * i] - lead_lag_path[0, 2 * i]) * (lead_lag_path[t + 1, 2 * j] - lead_lag_path[t, 2 * j])
    return S2

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1)}
    edges = [('A', 'B'), ('A', 'C')]
    root = 'A'
    path = np.random.rand(10, 2)
    print(hybrid_signature(path, nodes, edges, root))
    print(hybrid_path_signature(path, nodes, edges, root))
    print(hybrid_level2(path, nodes, edges, root))