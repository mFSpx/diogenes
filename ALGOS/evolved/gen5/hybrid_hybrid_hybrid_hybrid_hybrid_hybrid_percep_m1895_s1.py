# DARWIN HAMMER — match 1895, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2.py (gen3)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s2.py (gen4)
# born: 2026-05-29T23:39:31Z

import numpy as np
import math
import random
import sys
import pathlib
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
    dist: Dict[str, float] = {n: 0.0 for n in nodes}
    q: List[str] = [root]
    dist[root] = 0.0
    while q:
        n = q.pop(0)
        for e in edges:
            if e[0] == n:
                u = e[1]
                if u not in adj[n]:
                    adj[n].append(u)
                if dist[n] + length(nodes[n], nodes[u]) < dist[u]:
                    dist[u] = dist[n] + length(nodes[n], nodes[u])
                    q.append(u)
                edge_len[e] = length(nodes[n], nodes[u])
    return adj, edge_len, dist

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values)-1): bits = (bits<<1)|int(values[i] > values[i+1])
    return bits

def compute_phash(values: List[float]) -> int:
    if not values: return 0
    avg = sum(values) / len(values); bits = 0
    for v in values[:64]: bits = (bits<<1)|int(v>=avg)
    return bits

def hamming_distance(a: int, b: int) -> int: 
    return bin(a^b).count('1')

def similarity_matrix(features: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(features[ni])
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = gaussian(euclidean(features[ni], features[nj]))
                S[j, i] = S[i, j]
            elif j > i:
                S[i, j] = 0
                S[j, i] = S[i, j]
    return S, nodes

def hybrid_hygiene(node: str, adj: Dict[str, List[str]], edge_len: Dict[Tuple[str, str], float], dist: Dict[str, float], features: Dict[str, List[float]], S: np.ndarray, nodes: List[str]) -> float:
    """Hybrid decision hygiene scoring system."""
    node_features = features[node]
    similarity_scores = S[nodes.index(node)]
    hygiene_score = 0.0
    for neighbor in adj[node]:
        neighbor_features = features[neighbor]
        hygiene_score += gaussian(euclidean(node_features, neighbor_features)) * edge_len[(node, neighbor)]
    return hygiene_score + np.sum(similarity_scores)

def hybrid_tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    features: Dict[str, List[float]],
    S: np.ndarray,
    nodes: List[str],
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    S, _ = similarity_matrix(features)
    return adj, edge_len, dist

def hybrid_decision(node: str, adj: Dict[str, List[str]], edge_len: Dict[Tuple[str, str], float], dist: Dict[str, float], features: Dict[str, List[float]], S: np.ndarray, nodes: List[str]) -> Tuple[float, float]:
    """Hybrid decision system."""
    node_features = features[node]
    similarity_scores = S[nodes.index(node)]
    hygiene_score = hybrid_hygiene(node, adj, edge_len, dist, features, S, nodes)
    decision_score = 0.0
    for neighbor in adj[node]:
        neighbor_features = features[neighbor]
        decision_score += gaussian(euclidean(node_features, neighbor_features)) * edge_len[(node, neighbor)]
    return hygiene_score, decision_score

if __name__ == "__main__":
    nodes = { 'A': (0.0, 0.0), 'B': (1.0, 1.0), 'C': (2.0, 2.0) }
    edges = [ ('A', 'B'), ('B', 'C') ]
    root = 'A'
    features = { 'A': [1.0, 2.0], 'B': [3.0, 4.0], 'C': [5.0, 6.0] }
    S, nodes = similarity_matrix(features)
    adj, edge_len, dist = hybrid_tree_metrics(nodes, edges, root, features, S, nodes)
    hygiene_score, decision_score = hybrid_decision(root, adj, edge_len, dist, features, S, nodes)
    print(hygiene_score)
    print(decision_score)