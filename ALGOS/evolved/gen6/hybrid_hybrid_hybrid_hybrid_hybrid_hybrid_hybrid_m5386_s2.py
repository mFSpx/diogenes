# DARWIN HAMMER — match 5386, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1398_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2471_s0.py (gen5)
# born: 2026-05-30T00:01:41Z

"""
This module implements a novel hybrid algorithm that mathematically fuses the core topologies of two parent algorithms: 
'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1398_s2.py' and 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2471_s0.py'. 
The mathematical bridge lies in combining the stylometry feature extraction from the first parent with the tree metrics and work unit allocation from the second parent. 
The hybrid system uses the stylometry features to update the edge conductance in the tree, and then allocates work units based on the similarity to a prototype vector, 
using the SSIM score as a metric. The resulting hybrid system integrates the geometric quantities from the tree with the probabilistic weights from the stylometry features.
"""

import numpy as np
import math
import random
import sys
import pathlib

def stylometry_feature_vector(text_data: str) -> np.ndarray:
    words = text_data.split()
    feature_vector = np.zeros((len(words), 3))
    for i, word in enumerate(words):
        if word in ["i", "me", "my", "mine", "myself"]:
            feature_vector[i, 0] = 1
        if word in ["you", "your", "yours", "yourself"]:
            feature_vector[i, 1] = 1
        if word in ["he", "him", "his", "himself"]:
            feature_vector[i, 2] = 1
    return feature_vector

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

    dist: Dict[str, float] = {root: 0.0}
    for node in nodes:
        if node != root:
            dist[node] = float('inf')
    for _ in range(len(nodes)):
        for a, b in edges:
            if dist[a] != float('inf') and dist[a] + edge_len[(a, b)] < dist[b]:
                dist[b] = dist[a] + edge_len[(a, b)]
    return adj, edge_len, dist

def hybrid_conductance_update(conductance: np.ndarray, feature_vector: np.ndarray, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> np.ndarray:
    return np.maximum(0.0, conductance + dt * (gain * np.abs(feature_vector) - decay * conductance))

def allocate_workshare_ssim(
    x: np.ndarray, 
    y: np.ndarray, 
    nodes: Dict[str, Tuple[float, float]], 
    edges: List[Tuple[str, str]], 
    edge_len: Dict[Tuple[str, str], float],
    root: str,
    total_units: float = 1.0
):
    adj, _, dist = tree_metrics(nodes, edges, root)
    feature_vector = stylometry_feature_vector(" ".join([str(val) for val in x]))
    conductance = hybrid_conductance_update(np.ones(len(feature_vector)), feature_vector)
    ssim = np.mean(conductance)
    workshare = {}
    for node in nodes:
        workshare[node] = ssim * total_units / len(nodes)
    return workshare

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (0, 1),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    workshare = allocate_workshare_ssim(x, y, nodes, edges, {}, root)
    print(workshare)