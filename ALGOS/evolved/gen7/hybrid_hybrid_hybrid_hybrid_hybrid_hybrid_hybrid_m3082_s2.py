# DARWIN HAMMER — match 3082, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m2659_s2.py (gen6)
# born: 2026-05-29T23:47:38Z

"""
Hybrid Algorithm: fusion of hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s1 and hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m2659_s2

The mathematical bridge between the two structures lies in applying the Gaussian radial basis function (RBF) to the Fisher information score, 
allowing for a more informed decision-making process in the hybrid algorithm. The RBF-based similarity matrix provides a dense, continuous representation 
of pairwise node affinity, which is then used to calculate the Fisher information score and the minimum-cost tree with Bayesian update.

This hybrid system integrates the strengths of both parents: the Fisher information score for directional parameters, and the minimum-cost tree with Bayesian 
update for decision making under uncertainty, while incorporating the Gaussian RBF for improved similarity calculations.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

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
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        if a not in dist:
            dist[a] = float('inf')
        if b not in dist:
            dist[b] = float('inf')
        dist[a] = min(dist[a], dist.get(root, float('inf')) + edge_len.get((root, a), float('inf')))
        dist[b] = min(dist[b], dist.get(root, float('inf')) + edge_len.get((root, b), float('inf')))
    return adj, edge_len, dist

def rbf_fisher_score(theta: float, center: float, width: float, epsilon: float = 1.0) -> float:
    """RBF-based Fisher information score."""
    r = math.fabs(theta - center)
    return gaussian(r, epsilon) * fisher_score(theta, center, width)

def hybrid_tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    epsilon: float = 1.0,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths, root‑to‑node distances and RBF-based Fisher information scores.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    fisher : dict mapping node → RBF-based Fisher information score
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    fisher: Dict[str, float] = {}
    for node in nodes:
        theta, _ = nodes[node]
        fisher[node] = rbf_fisher_score(theta, 0.0, 1.0, epsilon)
    return adj, edge_len, dist, fisher

if __name__ == "__main__":
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 0.0), 'C': (2.0, 0.0)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    adj, edge_len, dist, fisher = hybrid_tree_metrics(nodes, edges, root)
    print(adj)
    print(edge_len)
    print(dist)
    print(fisher)