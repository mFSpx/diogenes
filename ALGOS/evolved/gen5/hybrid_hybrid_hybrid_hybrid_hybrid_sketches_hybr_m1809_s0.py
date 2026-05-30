# DARWIN HAMMER — match 1809, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py (gen3)
# parent_b: hybrid_sketches_hybrid_hybrid_hybrid_m302_s0.py (gen4)
# born: 2026-05-29T23:40:17Z

"""
This module fuses two parent algorithms:
- **hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py** – provides 
  tree construction, Euclidean edge lengths, and Bayesian marginal/posterior 
  calculations.
- **hybrid_sketches_hybrid_hybrid_hybrid_m302_s0.py** – provides count-min 
  sketch and B-spline basis functions for high-dimensional data encoding.

The mathematical bridge is the use of B-spline basis functions to encode the 
expected VRAM consumption of a model component tree, which is then used to 
update the Bayesian posterior probability distribution over the nodes. This 
distribution is then used to compute the Shannon entropy of the posterior, 
which is used to determine the decision-hygiene metrics.

The result is a single unified system that can advise whether the current 
model configuration fits within the available GPU memory, while also providing 
a way to encode and process high-dimensional data using count-min sketch and 
B-spline basis functions.
"""

import math
import random
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
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
    adjacency : Dict[str, List[str]]
        Adjacency list representation of the tree.
    edge_lengths : Dict[Tuple[str, str], float]
        Euclidean lengths of the edges in the tree.
    node_distances : Dict[str, float]
        Distances from the root node to each node in the tree.
    """
    adjacency = {}
    edge_lengths = {}
    node_distances = {}

    for edge in edges:
        if edge[0] not in adjacency:
            adjacency[edge[0]] = []
        if edge[1] not in adjacency:
            adjacency[edge[1]] = []
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
        edge_lengths[edge] = length(nodes[edge[0]], nodes[edge[1]])
        edge_lengths[(edge[1], edge[0])] = length(nodes[edge[0]], nodes[edge[1]])

    visited = set()
    stack = [root]
    node_distances[root] = 0.0

    while stack:
        node = stack.pop()
        visited.add(node)

        for neighbor in adjacency[node]:
            if neighbor not in visited:
                stack.append(neighbor)
                node_distances[neighbor] = node_distances[node] + edge_lengths[(node, neighbor)]

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

def count_min_sketch(items: List[str], width: int=64, depth: int=4) -> List[List[int]]:
    table = [[0]*width for _ in range(depth)]
    for item in items:
        for i in range(depth):
            hash_value = int(hashlib.sha256(item.encode()).hexdigest(), 16) % width
            table[i][hash_value] += 1
    return table

def hybrid_operation(node_sizes, prior_probabilities, likelihood_vector, grid, k=3):
    """
    Hybrid operation that fuses the Bayesian update and B-spline basis functions.

    Parameters
    ----------
    node_sizes : List[float]
        Sizes of the nodes in the tree.
    prior_probabilities : List[float]
        Prior probabilities of the nodes.
    likelihood_vector : List[float]
        Likelihood vector representing evidence.
    grid : List[float]
        Grid points for the B-spline basis functions.
    k : int, optional
        Order of the B-spline basis functions. Defaults to 3.

    Returns
    -------
    posterior_probabilities : List[float]
        Posterior probabilities of the nodes.
    expected_vram : float
        Expected VRAM consumption.
    entropy : float
        Shannon entropy of the posterior probabilities.
    """
    posterior_probabilities = [prior * likelihood for prior, likelihood in zip(prior_probabilities, likelihood_vector)]
    posterior_probabilities = [prob / sum(posterior_probabilities) for prob in posterior_probabilities]

    expected_vram = sum(node_size * posterior_probability for node_size, posterior_probability in zip(node_sizes, posterior_probabilities))

    entropy = -sum(posterior_probability * math.log(posterior_probability, 2) for posterior_probability in posterior_probabilities if posterior_probability > 0)

    x = np.array([node_size for node_size in node_sizes])
    B = bspline_basis(x, grid, k)
    encoded_vram = np.dot(B, np.array(posterior_probabilities))

    return posterior_probabilities, expected_vram, entropy, encoded_vram

if __name__ == "__main__":
    node_sizes = [1.0, 2.0, 3.0, 4.0]
    prior_probabilities = [0.25, 0.25, 0.25, 0.25]
    likelihood_vector = [0.5, 0.3, 0.1, 0.1]
    grid = [0.0, 1.0, 2.0, 3.0, 4.0]

    posterior_probabilities, expected_vram, entropy, encoded_vram = hybrid_operation(node_sizes, prior_probabilities, likelihood_vector, grid)

    print("Posterior Probabilities:", posterior_probabilities)
    print("Expected VRAM:", expected_vram)
    print("Entropy:", entropy)
    print("Encoded VRAM:", encoded_vram)