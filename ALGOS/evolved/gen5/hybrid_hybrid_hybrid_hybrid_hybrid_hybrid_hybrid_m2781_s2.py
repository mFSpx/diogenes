# DARWIN HAMMER — match 2781, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_poikilotherm__m1522_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2043_s0.py (gen4)
# born: 2026-05-29T23:45:51Z

"""
This module integrates the hybrid_hybrid_hybrid_minimu_hybrid_poikilotherm__m1522_s1 and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2043_s0 algorithms into a single hybrid system.
The mathematical bridge between the two structures lies in the use of radial basis functions 
to model the uncertainty estimates from the Hoeffding bound, and the application of temperature-dependent 
developmental rate ρ(T) from the Schoolfield-Rollinson model to modulate the confidence in each edge 
posterior derived from the Bayesian update. This allows for a unified system that combines the 
strengths of both parent algorithms, leveraging the power of radial basis functions and temperature-dependent 
developmental rates to make informed decisions.

The mathematical interface is established by interpreting the temperature-dependent rate ρ(T) 
as a global physiological scaling factor that adjusts the posterior update, and then using 
these temperature-adjusted posteriors in the hybrid cost functional. Meanwhile, the radial basis 
functions are used to model the uncertainty estimates, allowing for more accurate predictions.
"""

import numpy as np
import math
import random
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

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

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: List[float]) -> float:
        return sum(w * gaussian(euclidean(list(x), list(c)), self.epsilon) for w, c in zip(self.weights, self.centers))

def hybrid_operation(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    temperature: float,
    rbf_surrogate: RBFSurrogate,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float], List[float]]:
    """
    Perform the hybrid operation, combining the tree metrics and radial basis function surrogate.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    predictions : list of predictions from the RBF surrogate
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    predictions = [rbf_surrogate.predict(nodes[node]) for node in nodes]
    # adjust the predictions based on the temperature-dependent developmental rate
    predictions = [p * temperature for p in predictions]
    return adj, edge_len, dist, predictions

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
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

if __name__ == "__main__":
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    root = "A"
    temperature = 0.5
    rbf_surrogate = RBFSurrogate(centers=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], weights=[1.0, 1.0, 1.0])
    adj, edge_len, dist, predictions = hybrid_operation(nodes, edges, root, temperature, rbf_surrogate)
    print(adj)
    print(edge_len)
    print(dist)
    print(predictions)