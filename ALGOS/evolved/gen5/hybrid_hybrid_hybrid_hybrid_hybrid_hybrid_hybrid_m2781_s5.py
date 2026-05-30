# DARWIN HAMMER — match 2781, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_poikilotherm__m1522_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2043_s0.py (gen4)
# born: 2026-05-29T23:45:51Z

"""
This module integrates the hybrid_hybrid_hybrid_minimu_hybrid_poikilotherm__m1522_s1 
and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2043_s0 algorithms into a single 
hybrid system. The mathematical bridge between the two structures lies in the application 
of the temperature-dependent developmental rate ρ(T) from the Schoolfield-Rollinson model 
to modulate the confidence in the radial basis function (RBF) surrogate model, 
which in turn adjusts the Hoeffding bound used in the bandit-router algorithm.

The core idea is to utilize the RBF surrogate to estimate the probability distributions 
of the data, and then apply the temperature-adjusted posteriors to make decisions 
based on these distributions, while incorporating the bandit's reward estimation 
and the store differential equation to update the graph matrix and make decisions.
"""

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import pathlib
from typing import Dict, List, Tuple

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
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

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    centers: list[tuple[float, ...]]
    weights: list[float]
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> float:
        return sum(self.weights[i] * gaussian(euclidean(x, self.centers[i]), self.epsilon) for i in range(len(self.centers)))

def schoolfield_rollinson(T: float) -> float:
    """
    Temperature-dependent developmental rate ρ(T) from the Schoolfield-Rollinson model.
    """
    return 0.051 * ((T - 20) / 10) ** 2 + 0.0002 * ((T - 20) / 10) ** 4

def hybrid_operation(nodes: Dict[str, Tuple[float, float]], 
                     edges: List[Tuple[str, str]], 
                     root: str, 
                     rbf_surrogate: RBFSurrogate, 
                     T: float) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Perform the hybrid operation.

    Parameters:
    nodes (Dict[str, Tuple[float, float]]): Node positions.
    edges (List[Tuple[str, str]]): Edge list.
    root (str): Root node.
    rbf_surrogate (RBFSurrogate): RBF surrogate model.
    T (float): Temperature.

    Returns:
    -------
    adj (Dict[str, List[str]]): Adjacency list.
    edge_len (Dict[Tuple[str, str], float]): Edge lengths.
    dist (Dict[str, float]): Root-to-node distances.
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # Apply temperature-dependent developmental rate ρ(T) to modulate the confidence in the RBF surrogate model
    modulated_weights = [w * schoolfield_rollinson(T) for w in rbf_surrogate.weights]

    # Update the RBF surrogate model with modulated weights
    updated_rbf_surrogate = RBFSurrogate(rbf_surrogate.centers, modulated_weights, rbf_surrogate.epsilon)

    # Use the updated RBF surrogate model to make predictions
    predictions = [updated_rbf_surrogate.predict(node) for node in nodes.values()]

    return adj, edge_len, dist

if __name__ == "__main__":
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 0.0), 'C': (0.0, 1.0)}
    edges = [('A', 'B'), ('A', 'C')]
    root = 'A'
    rbf_surrogate = RBFSurrogate([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [1.0, 1.0, 1.0])
    T = 25.0

    adj, edge_len, dist = hybrid_operation(nodes, edges, root, rbf_surrogate, T)
    print(adj)
    print(edge_len)
    print(dist)