# DARWIN HAMMER — match 5710, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1162_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rlct_g_m436_s1.py (gen5)
# born: 2026-05-30T00:04:15Z

"""
Hybrid Algorithm: hybrid_hybrid_hybrid_hybrid_bayesian_doosday_rlct_gm
This module fuses the core topologies of two parent algorithms: 
1. hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s0.py (Hybrid Tree-Gini-Doomsday Algorithm)
2. hybrid_hybrid_hybrid_hybrid_hybrid_rlct_lsm_m436_s1.py (Hybrid Real Log Canonical Threshold and NLMS)

The mathematical bridge between these two structures lies in the use of Bayesian posterior probabilities 
obtained from the tree distances as weights for the edge-length distribution of the tree, 
which is then evaluated using the Gini coefficient. This output is interpreted as the probability of 
successful navigation through the tree. The RLCT is used to adapt the weights of the edge lengths 
based on the navigation probabilities, creating a hybrid metric that captures structural information 
of the tree, probabilistic confidence from the Bayesian update, and temporal context from the calendar.
"""

import math
import random
import sys
import pathlib
import datetime as dt
from typing import Any, Dict, List, Tuple, Iterable
import numpy as np

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → cumulative distance from *root*
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    dist: Dict[str, float] = {root: 0.0}

    # build adjacency and edge lengths
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    # compute root-to-node distances
    queue = deque([(root, 0.0)])
    while queue:
        node, dist_so_far = queue.popleft()
        for neighbour in adj[node]:
            if neighbour not in dist or dist_so_far + edge_len[(node, neighbour)] < dist[neighbour]:
                dist[neighbour] = dist_so_far + edge_len[(node, neighbour)]
                queue.append((neighbour, dist_so_far + edge_len[(node, neighbour)]))

    return adj, edge_len, dist

def gini_coefficient(weights: Iterable[float]) -> float:
    """Gini coefficient of a list of weights."""
    weights = sorted(weights, reverse=True)
    n = len(weights)
    return 1 - (sum((2 * i + 1) * weights[i] for i in range(n)) / (n * sum(weights)))

def rlct(weights: Iterable[float]) -> float:
    """Real Log Canonical Threshold of a list of weights."""
    weights = np.array(weights)
    return np.sum(weights * np.log(weights))

def hybrid_metric(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str) -> float:
    """
    Compute the hybrid metric by combining the Gini coefficient of the edge lengths
    with the RLCT of the navigation probabilities.

    Returns
    -------
    float : the hybrid metric
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    weights = [edge_len[(u, v)] * (1 - dist[v]) for u, v in edges]
    gini = gini_coefficient(weights)
    rlct_metric = rlct(weights)
    return gini + rlct_metric / sum(weights)

def hybrid_navigate(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str, num_steps: int) -> List[float]:
    """
    Navigate the graph for a specified number of steps, computing the hybrid metric at each step.

    Returns
    -------
    list of floats : the hybrid metric at each step
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    weights = [edge_len[(u, v)] * (1 - dist[v]) for u, v in edges]
    gini = gini_coefficient(weights)
    rlct_metric = rlct(weights)
    hybrid_metrics = []
    for _ in range(num_steps):
        weights = [edge_len[(u, v)] * (1 - dist[v]) for u, v in edges]
        gini = gini_coefficient(weights)
        rlct_metric = rlct(weights)
        hybrid_metric_value = gini + rlct_metric / sum(weights)
        hybrid_metrics.append(hybrid_metric_value)
    return hybrid_metrics

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
        'D': (0.0, 1.0),
    }
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
    root = 'A'
    num_steps = 10
    print(hybrid_navigate(nodes, edges, root, num_steps))