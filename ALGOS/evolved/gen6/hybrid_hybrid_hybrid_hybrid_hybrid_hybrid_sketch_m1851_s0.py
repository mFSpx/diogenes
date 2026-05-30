# DARWIN HAMMER — match 1851, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py (gen5)
# born: 2026-05-29T23:39:08Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of 
the hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py and 
hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py algorithms. 
The mathematical bridge between the two structures lies in the incorporation of 
the count-min sketch from the hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py 
into the fisher score calculation of the hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py's 
Gaussian beam intensity calculation. This allows for efficient, probabilistic estimation of 
modulation vectors based on hashed item frequencies, which are then used to inform the Fisher score calculation.
"""

import math
import random
import sys
import pathlib
from collections import Counter
import numpy as np
import hashlib

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


def count_min_sketch(items: list[str], width: int=64, depth: int=4) -> list[list[int]]:
    table=[[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table


def hybrid_fisher_score(theta: float, center: float, width: float, items: list[str]) -> float:
    """
    Calculate the Fisher score incorporating the count-min sketch.
    
    The count-min sketch is used to estimate the frequency of each item, 
    which is then used to inform the Fisher score calculation.
    """
    count_min_table = count_min_sketch(items)
    estimated_frequencies = np.mean(count_min_table, axis=0)
    intensity = gaussian_beam(theta, center, width)
    derivative = intensity * (-(theta - center) / (width * width))
    fisher_score = (derivative * derivative) / intensity
    return fisher_score * np.mean(estimated_frequencies)


def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float], dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    edge_len: dict[tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: dict[str, float] = {root: 0.0}
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


def hybrid_tree_metrics(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
    items: list[str],
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float], dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances, 
    incorporating the count-min sketch.
    
    The count-min sketch is used to estimate the frequency of each item, 
    which is then used to inform the tree metrics calculation.
    """
    count_min_table = count_min_sketch(items)
    estimated_frequencies = np.mean(count_min_table, axis=0)
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    return adj, edge_len, {node: dist[node] * np.mean(estimated_frequencies) for node in dist}


if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
    }
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    items = ['item1', 'item2', 'item3']
    adj, edge_len, dist = hybrid_tree_metrics(nodes, edges, root, items)
    print(adj)
    print(edge_len)
    print(dist)