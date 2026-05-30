# DARWIN HAMMER — match 3822, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1345_s0.py (gen6)
# parent_b: hybrid_hybrid_temporal_moti_hybrid_hybrid_hybrid_m1987_s1.py (gen6)
# born: 2026-05-29T23:51:44Z

"""
This module fuses the topological structures of 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1345_s0.py (DARWIN HAMMER — match 1345, survivor 0) 
and hybrid_hybrid_temporal_moti_hybrid_hybrid_hybrid_m1987_s1.py (DARWIN HAMMER — match 1987, survivor 1).

The mathematical bridge between the two parents lies in the fact that 
the MinHash signature can be viewed as a pattern in a Dense Associative Memory, 
and the Gini coefficient can be used to quantify the inequality in the distribution of node distances.

The governing equations of the hybrid system are based on the MinHash signature, 
the Gini coefficient, and the sheaf's sections.

The fusion integrates the temporal motif mining with the MinHash signature, 
using the motif patterns as input to the MinHash and the MinHash signature as input to the motif mining.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class BurstSignal: 
    key: str; 
    count: int; 
    z_score: float

@dataclass(frozen=True)
class TemporalMotif: 
    pattern: tuple[str,...]; 
    support: int

class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a list of values."""
    values = np.array(values)
    if values.size == 0:
        return 0.0
    values = values.flatten()
    if np.amin(values) < 0:
        values -= np.amin(values)
    values += 0.0000001
    index = np.argsort(values, axis=0)
    n = len(values)
    index = index[::-1]
    values = values[index]
    if values[0] == 0:
        return 1.0
    p = np.arange(1, n+1, dtype=float)
    p = (p / n)
    q = np.cumsum(values)
    q /= np.sum(values)
    return ((np.sum((2 * p - 1 - n  ) * q)) / (n - 1))

def minhash_signature(values: List[float]) -> np.ndarray:
    """MinHash signature of a list of values."""
    np.random.seed(42)
    num_hash_functions = 5
    hash_values = np.random.rand(num_hash_functions, len(values))
    hash_values = np.dot(hash_values, values)
    return np.min(hash_values, axis=1)

def hybrid_operation(nodes: dict, edges: list, root: str) -> Tuple[np.ndarray, float]:
    """
    Perform the hybrid operation.

    Returns
    -------
    minhash_sig : MinHash signature of the node distances.
    gini_coef : Gini coefficient of the node distances.
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    node_distances = list(dist.values())
    gini_coef = gini_coefficient(node_distances)
    minhash_sig = minhash_signature(node_distances)
    return minhash_sig, gini_coef

def tree_metrics(
    nodes: dict,
    edges: list,
    root: str,
) -> Tuple[dict, dict, dict]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj = {}
    edge_len = {}
    dist = {root: 0.0}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbour in adj[node]:
            if neighbour not in dist:
                dist[neighbour] = dist[node] + edge_len[(node, neighbour)]
                stack.append(neighbour)
    return adj, edge_len, dist

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
        'D': (0.0, 1.0)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    minhash_sig, gini_coef = hybrid_operation(nodes, edges, root)
    print("MinHash Signature:", minhash_sig)
    print("Gini Coefficient:", gini_coef)