# DARWIN HAMMER — match 1851, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py (gen5)
# born: 2026-05-29T23:39:08Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of 
hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py and 
hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py.

The mathematical bridge between the two structures lies in the incorporation of 
the Fisher information from hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py 
into the modulation_vector generation of hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py's 
RBF surrogate. This allows for efficient, probabilistic estimation of 
modulation vectors based on hashed item frequencies and Fisher information.

The governing equations of the hybrid algorithm are based on the Fisher information 
for a Gaussian beam and the count-min sketch data structure.

Parent algorithms:
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s2.py
- hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s2.py
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
import hashlib
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


def count_min_sketch(items: list[str], width: int=64, depth: int=4) -> list[list[int]]:
    table=[[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): 
            table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table


def symbol_vector(symbol: str, dim: int = 10000) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def bind(a: list[int], b: list[int]) -> list[int]:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: list[list[int]]) -> list[int]:
    if not vectors:
        raise ValueError('at least one vector is required')
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError('all vectors must have equal length')
    return [sum(x) for x in zip(*vectors)]


def hybrid_modulation_vector(items: list[str], theta: float, center: float, width: float) -> list[int]:
    sketch = count_min_sketch(items)
    modulation_vector = []
    for row in sketch:
        symbol = ''.join(map(str, row))
        vector = symbol_vector(symbol)
        fisher_info = fisher_score(theta, center, width)
        modulation_vector.append(bind(vector, [int(fisher_info)] * len(vector)))
    return bundle(modulation_vector)


def hybrid_tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    items: list[str],
    theta: float,
    center: float,
    width: float
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float], list[int]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.
    Also compute a hybrid modulation vector.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    modulation_vector : list[int]
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    modulation_vector = hybrid_modulation_vector(items, theta, center, width)
    return adj, edge_len, dist, modulation_vector


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
        edge_len[(a, b)] = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
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


if __name__ == "__main__":
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 0.0), 'C': (0.0, 1.0)}
    edges = [('A', 'B'), ('A', 'C')]
    root = 'A'
    items = ['item1', 'item2', 'item3']
    theta = 0.5
    center = 0.0
    width = 1.0
    adj, edge_len, dist, modulation_vector = hybrid_tree_metrics(nodes, edges, root, items, theta, center, width)
    print(adj)
    print(edge_len)
    print(dist)
    print(modulation_vector)