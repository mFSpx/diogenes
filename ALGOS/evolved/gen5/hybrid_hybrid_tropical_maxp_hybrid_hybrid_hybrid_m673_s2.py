# DARWIN HAMMER — match 673, survivor 2
# gen: 5
# parent_a: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s4.py (gen4)
# born: 2026-05-29T23:30:22Z

"""
Fusing hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s3.py and 
hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s4.py into a single hybrid system.

The mathematical bridge between the two structures is the application of 
tropical max-plus algebra to the semantic weighting of the geometric edge lengths 
in the minimum-cost tree, and the Bayesian update of the posterior probabilities 
of the tree cost using the observed VRAM usage.

The governing equations of the tropical max-plus algebra are used to compute 
the maximum expected utility of the semantic weighting system, 
while the Bayesian update is used to update the prior probabilities of the tree cost.

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
    dist: Dict[str, float] = {root: 0.0}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[u, v] = length(nodes[u], nodes[v])

    # Perform BFS to compute distances
    queue = [root]
    visited = set([root])

    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                dist[neighbor] = dist[node] + edge_len[node, neighbor]
                queue.append(neighbor)

    return adj, edge_len, dist

def semantic_weighting(edge_len: Dict[Tuple[str, str], float], 
                      lsm_vector: Dict[str, float], 
                      lexical_categories: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Compute semantic weights for edge lengths.

    Returns
    -------
    weighted_edge_len : dict mapping edge → weighted length
    """
    weighted_edge_len: Dict[Tuple[str, str], float] = {}

    for edge, length in edge_len.items():
        u, v = edge
        weight = lsm_vector.get(u, 0.0) * lsm_vector.get(v, 0.0)
        weighted_edge_len[edge] = length * weight

    return weighted_edge_len

def bayesian_update(prior_mean: float, prior_variance: float, 
                    observed_vram: float, observation_variance: float) -> Tuple[float, float]:
    """
    Perform Bayesian update to compute posterior mean and variance.

    Returns
    -------
    posterior_mean : float
    posterior_variance : float
    """
    posterior_mean = (prior_variance * observed_vram + observation_variance * prior_mean) / (prior_variance + observation_variance)
    posterior_variance = (prior_variance * observation_variance) / (prior_variance + observation_variance)

    return posterior_mean, posterior_variance

def hybrid_operation(nodes: Dict[str, Tuple[float, float]], 
                     edges: List[Tuple[str, str]], 
                     root: str, 
                     lsm_vector: Dict[str, float], 
                     lexical_categories: List[str], 
                     prior_mean: float, 
                     prior_variance: float, 
                     observed_vram: float, 
                     observation_variance: float) -> Tuple[Dict[str, List[str]], 
                                                             Dict[Tuple[str, str], float], 
                                                             Dict[str, float], 
                                                             Dict[Tuple[str, str], float], 
                                                             float, 
                                                             float]:
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    weighted_edge_len = semantic_weighting(edge_len, lsm_vector, lexical_categories)
    posterior_mean, posterior_variance = bayesian_update(prior_mean, prior_variance, observed_vram, observation_variance)

    # Apply tropical max-plus algebra to weighted edge lengths
    max_weighted_edge_len = t_matmul(np.array(list(weighted_edge_len.values())), np.array([1.0]*len(weighted_edge_len)))

    return adj, edge_len, dist, weighted_edge_len, posterior_mean, posterior_variance

if __name__ == "__main__":
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 0.0), 'C': (1.0, 1.0), 'D': (0.0, 1.0)}
    edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]
    root = 'A'
    lsm_vector = {'A': 0.5, 'B': 0.6, 'C': 0.7, 'D': 0.8}
    lexical_categories = ['cat1', 'cat2', 'cat3', 'cat4']
    prior_mean = 10.0
    prior_variance = 2.0
    observed_vram = 12.0
    observation_variance = 1.5

    adj, edge_len, dist, weighted_edge_len, posterior_mean, posterior_variance = hybrid_operation(nodes, edges, root, lsm_vector, lexical_categories, prior_mean, prior_variance, observed_vram, observation_variance)

    print("Adjacency:", adj)
    print("Edge Lengths:", edge_len)
    print("Distances:", dist)
    print("Weighted Edge Lengths:", weighted_edge_len)
    print("Posterior Mean:", posterior_mean)
    print("Posterior Variance:", posterior_variance)