# DARWIN HAMMER — match 4054, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s5.py (gen4)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_decisi_m1508_s0.py (gen4)
# born: 2026-05-29T23:53:22Z

"""
Module hybrid_hybrid_fusion_m1186_m1508.py:
This module fuses the tree-based optimization from 
hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s5.py (PARENT ALGORITHM A) 
with the radial-basis surrogate model from 
hybrid_hybrid_perceptual_de_hybrid_hybrid_decisi_m1508_s0.py (PARENT ALGORITHM B).

The mathematical bridge between the two structures lies in the use of 
radial basis functions to model the material and path costs in the 
tree-based optimization, effectively creating a probabilistic surrogate 
model for optimization with enhanced robustness to noise.

The governing equations of PARENT ALGORITHM A are based on 
tree traversals and material/path costs, while PARENT ALGORITHM B 
uses radial basis functions for surrogate modeling. The fusion 
integrates these two approaches by using RBFs to model the material 
and path costs from PARENT ALGORITHM A, allowing for more robust 
and flexible optimization.
"""

import math
import numpy as np
import random
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import pathlib

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Compute Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class Point:
    x: float
    y: float

def tree_cost(
    nodes: dict[str, Point],
    edges: list[tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
    epsilon: float = 1.0
) -> float:
    """Total cost = material + weighted path‑to‑root cost, modeled with RBFs."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        dist = euclidean((nodes[u].x, nodes[u].y), (nodes[v].x, nodes[v].y))
        material += gaussian(dist, epsilon)

    # BFS from root to compute distances
    dist: dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + euclidean((nodes[cur].x, nodes[cur].y), (nodes[nxt].x, nodes[nxt].y))
                stack.append(nxt)

    path_cost = sum(gaussian(dist[v], epsilon) for v in dist)
    return material + path_weight * path_cost

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve linear system."""
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

def hybrid_optimization(
    nodes: dict[str, Point],
    edges: list[tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
    epsilon: float = 1.0
) -> list[float]:
    """Hybrid optimization using RBFs and tree traversals."""
    tree_cost_value = tree_cost(nodes, edges, root, path_weight, epsilon)
    # Create a linear system to optimize
    A = [[1.0, 2.0], [3.0, 4.0]]
    b = [tree_cost_value, tree_cost_value * 2.0]
    return solve_linear(A, b)

if __name__ == "__main__":
    nodes = {"A": Point(0.0, 0.0), "B": Point(1.0, 1.0), "C": Point(2.0, 2.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    result = hybrid_optimization(nodes, edges, root)
    print(result)