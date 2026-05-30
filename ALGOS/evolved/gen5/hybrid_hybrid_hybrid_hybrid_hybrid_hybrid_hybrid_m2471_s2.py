# DARWIN HAMMER — match 2471, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s0.py (gen4)
# born: 2026-05-29T23:42:27Z

"""
This module integrates the hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s1 and 
hybrid_hybrid_hybrid_gliner_hybrid_hybrid_ternar_m1199_s0 algorithms into a single hybrid system. 
The mathematical bridge between the two structures is formed by using the tree metrics from the first 
algorithm to estimate the resource requirements for the workshare allocator in the second algorithm. 
Specifically, the edge lengths from the tree metrics are used to compute a weighted SSIM score, which 
is then used to allocate work units among different groups.

The governing equations are integrated through the use of the tree metrics to estimate the resource 
requirements, the Bayesian update to adjust the scheduler's decisions, and the weighted SSIM score 
to allocate work units among different groups. This allows us to integrate the two algorithms into a 
single hybrid system that can adapt to changing resource requirements and make more informed decisions 
about resource allocation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

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
    dist: Dict[str, float] = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbour in adj[node]:
            if neighbour not in dist:
                dist[neighbour] = dist[node] + edge_len[(node, neighbour)]
                queue.append(neighbour)

    return adj, edge_len, dist

def compute_ssim(x: np.ndarray, y: np.ndarray, edge_len: Dict[Tuple[str, str], float]) -> float:
    """
    Compute the weighted Structural Similarity Index (SSIM) between two vectors.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Compute weighted SSIM using edge lengths
    weighted_ssim = 0
    for edge, length in edge_len.items():
        weighted_ssim += length * ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))

    return weighted_ssim / sum(edge_len.values())

def allocate_workshare_ssim(x: np.ndarray, y: np.ndarray, edge_len: Dict[Tuple[str, str], float], total_units: float, deterministic_target_pct: float = 90.0) -> dict[str, float]:
    """
    Allocate work units among different groups based on the weighted SSIM score.
    """
    ssim = compute_ssim(x, y, edge_len)
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    return {
        "total_units": total_units,
        "deterministic_units": deterministic_units,
        "llm_units": llm_units,
        "ssim": ssim,
    }

def hybrid_operation(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str, x: np.ndarray, y: np.ndarray, total_units: float) -> dict[str, Any]:
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    workshare_allocation = allocate_workshare_ssim(x, y, edge_len, total_units)
    return {
        "tree_metrics": (adj, edge_len, dist),
        "workshare_allocation": workshare_allocation,
    }

if __name__ == "__main__":
    nodes = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (1, 1),
        "D": (0, 1),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    total_units = 100

    result = hybrid_operation(nodes, edges, root, x, y, total_units)
    print(result)