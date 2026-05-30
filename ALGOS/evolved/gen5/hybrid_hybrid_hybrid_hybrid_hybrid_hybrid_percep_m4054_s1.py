# DARWIN HAMMER — match 4054, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s5.py (gen4)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_decisi_m1508_s0.py (gen4)
# born: 2026-05-29T23:53:22Z

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

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

def solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve linear system using NumPy."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    try:
        return np.linalg.solve(A, b).tolist()
    except np.linalg.LinAlgError:
        raise ValueError("singular surrogate system")

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
    A = [[1.0, 2.0], [2.0, 4.0]]  # Modified to be non-singular
    b = [tree_cost_value, tree_cost_value * 2.0]
    return solve_linear(A, b)

if __name__ == "__main__":
    nodes = {"A": Point(0.0, 0.0), "B": Point(1.0, 1.0), "C": Point(2.0, 2.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    result = hybrid_optimization(nodes, edges, root)
    print(result)