# DARWIN HAMMER — match 190, survivor 3
# gen: 3
# parent_a: tropical_maxplus.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# born: 2026-05-29T23:27:29Z

"""
Fusing tropical_maxplus.py and hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py into a single hybrid system.
The mathematical bridge between the two structures is the application of tropical max-plus algebra 
to the decision hygiene scoring system of the hybrid decision algorithm, 
and the expected cost of the minimum-cost tree computed using Bayesian update.

The governing equations of the tropical max-plus algebra are used to compute 
the maximum expected utility of the decision hygiene scoring system, 
while the Bayesian update is used to update the prior probabilities of the minimum-cost tree.

This hybrid system integrates the core topologies of both parent algorithms 
into a unified system, enabling the computation of maximum expected utility 
and posterior probabilities simultaneously.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

def t_add(x, y):
    """Tropical addition: max(x, y). Broadcasts."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication: x + y. Broadcasts."""
    return np.add(x, y)

def t_matmul(A, B):
    """Tropical matrix multiply.

    C[i, j] = max_k( A[i, k] + B[k, j] )

    A: (m, p), B: (p, n) → C: (m, n).
    Handles -inf entries correctly via numpy broadcasting.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcast: A[i, k, newaxis] + B[newaxis, k, j] then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

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
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P
    """
    return likelihood * prior / marginal

def hybrid_decision_tropical(nodes: Dict[str, Tuple[float, float]], 
                             edges: List[Tuple[str, str]], 
                             root: str, 
                             prior: np.ndarray, 
                             likelihood: np.ndarray, 
                             false_positive: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute maximum expected utility and posterior probabilities using tropical max-plus algebra 
    and Bayesian update.

    Returns
    -------
    max_expected_utility : np.ndarray
    posterior_probabilities : np.ndarray
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # Compute maximum expected utility using tropical max-plus algebra
    max_expected_utility = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        max_expected_utility[i] = t_add(dist[node], -np.inf)

    # Compute posterior probabilities using Bayesian update
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior_probabilities = bayes_update(prior, likelihood, marginal)

    return max_expected_utility, posterior_probabilities

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1)}
    edges = [('A', 'B'), ('A', 'C')]
    root = 'A'
    prior = np.array([0.2, 0.3, 0.5])
    likelihood = np.array([0.6, 0.7, 0.8])
    false_positive = 0.1

    max_expected_utility, posterior_probabilities = hybrid_decision_tropical(nodes, edges, root, prior, likelihood, false_positive)
    print("Maximum Expected Utility:", max_expected_utility)
    print("Posterior Probabilities:", posterior_probabilities)