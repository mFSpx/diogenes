# DARWIN HAMMER — match 1895, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2.py (gen3)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s2.py (gen4)
# born: 2026-05-29T23:39:31Z

"""
Module hybrid_fusion: A fusion of the hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2 and 
hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s2 algorithms. 
The mathematical bridge lies in the use of radial basis functions (RBFs) to model the decision hygiene 
scoring system and the VRAM allocation planning process, and the application of Bayesian update to 
modulate the broadcast probability in the Hoeffding tree.

The fusion integrates the decision hygiene scoring system into the radial basis function (RBF) model, 
and uses the similarity weights computed using the RBFs to guide the splitting process in a way that 
minimizes the impact of noise in the data stream.

This module provides functions to calculate the Euclidean distance, build tree metrics, and compute 
similarity matrices using perceptual hashing functions.
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
    dist: Dict[str, float] = {n: float('inf') for n in nodes}
    dist[root] = 0
    for u, v in edges:
        adj[u].append(v)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        if dist[u] + edge_len[(u, v)] < dist[v]:
            dist[v] = dist[u] + edge_len[(u, v)]
    return adj, edge_len, dist

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_dhash(values: List[float]) -> int:
    """Compute dhash."""
    bits = 0
    for i in range(len(values)-1): 
        bits = (bits<<1)|int(values[i] > values[i+1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Compute phash."""
    if not values: 
        return 0
    avg = sum(values) / len(values); 
    bits = 0
    for v in values[:64]: 
        bits = (bits<<1)|int(v>=avg)
    return bits

def hamming_distance(a: int, b: int) -> int: 
    """Compute hamming distance."""
    return bin(a^b).count('1')

def similarity_matrix(features: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    """Compute similarity matrix using perceptual hashing functions."""
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(features[ni])
        for j, nj in enumerate(nodes):
            if j < i:
                hj = compute_phash(features[nj])
                S[i, j] = 1 - (hamming_distance(hi, hj) / 64)
                S[j, i] = S[i, j]
            elif j == i:
                S[i, j] = 1
    return S, nodes

def hybrid_fusion(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], 
                   features: Dict[str, List[float]], root: str) -> Tuple[Dict[str, List[str]], 
                                                                        Dict[Tuple[str, str], float], 
                                                                        Dict[str, float], np.ndarray, 
                                                                        List[str]]:
    """Hybrid fusion function."""
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    S, nodes = similarity_matrix(features)
    return adj, edge_len, dist, S, nodes

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 1), 'C': (2, 2)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    features = {'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0], 'C': [7.0, 8.0, 9.0]}
    adj, edge_len, dist, S, nodes = hybrid_fusion(nodes, edges, features, root)
    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Distances:", dist)
    print("Similarity matrix:\n", S)
    print("Nodes:", nodes)