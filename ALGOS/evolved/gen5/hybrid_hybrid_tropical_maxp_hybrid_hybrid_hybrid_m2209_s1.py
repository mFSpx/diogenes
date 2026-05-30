# DARWIN HAMMER — match 2209, survivor 1
# gen: 5
# parent_a: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s3.py (gen4)
# born: 2026-05-29T23:41:28Z

import numpy as np
import math
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple, Sequence

# ----------------------------------------------------------------------
# Tropical max‑plus primitives (from Parent A)
# ----------------------------------------------------------------------
def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matvec_mul(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.max(A + v, axis=1)

# ----------------------------------------------------------------------
# Tree utilities (from Parent A)
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    dist: Dict[str, float] = {root: 0.0}
    queue: List[str] = [root]
    visited = {root}
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                dist[nb] = dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                queue.append(nb)

    return adj, edge_len, dist

# ----------------------------------------------------------------------
# Lead‑lag transform and weekday weighting (from Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: Sequence[Sequence[float]]) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D sequence")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

# ----------------------------------------------------------------------
# Simple B‑spline basis (lightweight version of Parent B)
# ----------------------------------------------------------------------
def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)

    n_basis = len(grid) - k - 1
    if n_basis <= 0:
        raise ValueError("grid too short for the requested order")

    knots = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))

    B = np.zeros((len(x), len(knots) - 1))
    for i in range(len(knots) - 1):
        left = knots[i]
        right = knots[i + 1]
        B[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    B[np.isclose(x, knots[-1]), -1] = 1.0  

    for d in range(1, k + 1):
        B_next = np.zeros((len(x), len(knots) - d - 1))
        for i in range(len(knots) - d - 1):
            denom1 = knots[i + d] - knots[i]
            term1 = 0.0 if denom1 == 0 else ((x - knots[i]) / denom1) * B[:, i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            term2 = 0.0 if denom2 == 0 else ((knots[i + d + 1] - x) / denom2) * B[:, i + 1]
            B_next[:, i] = term1 + term2
        B = B_next
    return B[:, :n_basis]

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def fused_edge_utility(
    edge_lengths: np.ndarray,
    groups: Sequence[str],
    dow: int,
    spline_grid: np.ndarray,
    spline_order: int = 3,
) -> np.ndarray:
    B = bspline_basis(edge_lengths, spline_grid, k=spline_order)  
    w = weekday_weight_vector(groups, dow)                    
    weighted_B = np.multiply(B, w)
    utilities = np.apply_along_axis(lambda x: t_add(np.sum(x), 0), 1, weighted_B)
    return utilities

def improved_fused_edge_utility(
    edge_lengths: np.ndarray,
    groups: Sequence[str],
    dow: int,
    spline_grid: np.ndarray,
    spline_order: int = 3,
) -> np.ndarray:
    B = bspline_basis(edge_lengths, spline_grid, k=spline_order)  
    w = weekday_weight_vector(groups, dow)                    
    weighted_B = np.multiply(B, w)
    utilities = np.apply_along_axis(lambda x: np.max(x), 1, weighted_B)
    return utilities

def main():
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
        'D': (0, 1),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    edge_lengths = np.array(list(edge_len.values()))
    groups = ['A', 'B', 'C', 'D']
    dow = 0
    spline_grid = np.array([0, 1, 2, 3])
    print(improved_fused_edge_utility(edge_lengths, groups, dow, spline_grid))

if __name__ == "__main__":
    main()