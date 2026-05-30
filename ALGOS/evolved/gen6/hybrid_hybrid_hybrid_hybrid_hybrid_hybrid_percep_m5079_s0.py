# DARWIN HAMMER — match 5079, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1895_s0.py (gen5)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_hybrid_m1188_s1.py (gen4)
# born: 2026-05-29T23:59:35Z

"""
Module hybrid_fusion: A fusion of the hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2 and 
hybrid_hybrid_perceptual_de_hybrid_hybrid_hybrid_m1188_s1 algorithms. 
The mathematical bridge lies in the integration of the radial basis function (RBF) model 
from the first parent with the pheromone-based decay model from the second parent, where 
the RBF model is used to compute the similarity weights between data points, and the 
pheromone decay model is used to update the weights of the RBF model. The perceptual 
hash functions are used to select the most representative data points for the RBF model.
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

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_dhash(values: list[float]) -> int:
    """Compute difference hash."""
    bits = 0
    for i in range(len(values) - 1): 
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: list[float]) -> int:
    """Compute perceptual hash."""
    if not values: 
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]: 
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int: 
    """Compute Hamming distance."""
    return (a ^ b).bit_count()

def cluster_by_phash(hashes: dict[str, int], max_distance: int = 4) -> list[list[str]]:
    """Cluster by perceptual hash."""
    clusters = []
    for k, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance: 
                c.append(k)
                break
        else: 
            clusters.append([k])
    return clusters

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
    dist: Dict[str, float] = {n: float('inf') for n in nodes}
    dist[root] = 0
    for u, v in edges:
        adj[u].append(v)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        if dist[u] + edge_len[(u, v)] < dist[v]:
            dist[v] = dist[u] + edge_len[(u, v)]
    return adj, edge_len, dist

def hybrid_fusion(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    values: list[float]
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float], int]:
    """Hybrid fusion function."""
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    phash = compute_phash(values)
    return adj, edge_len, dist, phash

def update_weights(
    adj: Dict[str, List[str]],
    edge_len: Dict[Tuple[str, str], float],
    dist: Dict[str, float],
    phash: int
) -> Dict[str, float]:
    """Update weights function."""
    weights = {}
    for node in adj:
        weights[node] = gaussian(dist[node])
    return weights

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (3, 4),
        'C': (6, 8)
    }
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    values = [1, 2, 3, 4, 5]
    adj, edge_len, dist, phash = hybrid_fusion(nodes, edges, root, values)
    weights = update_weights(adj, edge_len, dist, phash)
    print(adj)
    print(edge_len)
    print(dist)
    print(phash)
    print(weights)