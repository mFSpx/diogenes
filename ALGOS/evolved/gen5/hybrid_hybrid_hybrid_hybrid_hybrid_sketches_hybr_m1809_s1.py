# DARWIN HAMMER — match 1809, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py (gen3)
# parent_b: hybrid_sketches_hybrid_hybrid_hybrid_m302_s0.py (gen4)
# born: 2026-05-29T23:40:17Z

"""
Hybrid algorithm fusing 
- **hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py** 
  (Bayesian tree cost integration and VRAM scheduling) 
- **hybrid_sketches_hybrid_hybrid_hybrid_m302_s0.py** 
  (count-min sketch and B-spline basis functions).

The mathematical interface is established through the use of 
expected VRAM consumption as a prior for the count-min sketch, 
encoding model components as hashed items. 
B-spline basis functions project these items into a 
high-dimensional space for efficient similarity search.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
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
    adjacency: Dict[str, List[str]]
    edge_lengths: Dict[Tuple[str, str], float]
    node_distances: Dict[str, float]
    """
    adjacency = {node: [] for node in nodes}
    edge_lengths = {}
    node_distances = {root: 0}

    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        edge_lengths[(u, v)] = length(nodes[u], nodes[v])
        edge_lengths[(v, u)] = edge_lengths[(u, v)]

    # Perform BFS to compute node distances
    queue = [root]
    visited = set([root])

    while queue:
        node = queue.pop(0)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                node_distances[neighbor] = node_distances[node] + edge_lengths[(node, neighbor)]
                queue.append(neighbor)
                visited.add(neighbor)

    return adjacency, edge_lengths, node_distances

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    assert B.shape == (N, n_basis), (
        f"basis shape mismatch: got {B.shape}, expected ({N}, {n_basis})"
    )

    return B

def count_min_sketch(items: List[str], width: int=64, depth: int=4) -> list[list[int]]:
    table=[[0]*width for _ in range(depth)]
    for item in items:
        hash_values = [hash(item + str(i)) % width for i in range(depth)]
        for i, hash_value in enumerate(hash_values):
            table[i][hash_value] += 1
    return table

def hybrid_fusion(nodes: Dict[str, Tuple[float, float]], 
                  edges: List[Tuple[str, str]], 
                  root: str, 
                  items: List[str]):
    """
    Fuses Bayesian tree cost integration and count-min sketch.

    Returns
    -------
    expected_vram: float
    sketch: list[list[int]]
    """
    adjacency, edge_lengths, node_distances = tree_metrics(nodes, edges, root)

    # Compute prior probabilities for each node
    prior_probabilities = {node: 1 / len(nodes) for node in nodes}

    # Compute expected VRAM consumption
    node_sizes = {node: 1.0 for node in nodes}  # Assume unit size for each node
    expected_vram = sum(prior_probabilities[node] * node_sizes[node] for node in nodes)

    # Encode model components as hashed items
    hashed_items = [hash(item) for item in items]

    # Apply count-min sketch
    sketch = count_min_sketch(items)

    # Project hashed items into high-dimensional space using B-spline basis functions
    grid = np.linspace(0, 1, 100)
    projected_items = []
    for item in hashed_items:
        x = item / (2**32 - 1)  # Normalize hash value to [0, 1]
        basis = bspline_basis(x, grid)
        projected_items.append(basis)

    return expected_vram, sketch, projected_items

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (1, 1), 'D': (0, 1)}
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
    root = 'A'
    items = ['model_component_1', 'model_component_2', 'model_component_3']

    expected_vram, sketch, projected_items = hybrid_fusion(nodes, edges, root, items)
    print("Expected VRAM:", expected_vram)
    print("Count-min Sketch:")
    for row in sketch:
        print(row)
    print("Projected Items:")
    for item in projected_items:
        print(item)