# DARWIN HAMMER — match 2209, survivor 0
# gen: 5
# parent_a: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s3.py (gen4)
# born: 2026-05-29T23:41:28Z

"""
Hybrid Algorithm: tropical_maxplus_bspline_path

This module fuses the two parent algorithms:

* **Parent A** – tropical max‑plus algebra with tree metrics.
* **Parent B** – lead‑lag transform, weekday weight vector and B‑spline basis.

**Mathematical bridge**

1. The edge lengths of a tree (Parent A) are interpreted as “knots” on which a
   B‑spline basis (Parent B) is evaluated.
2. The resulting spline coefficients are multiplied (tropical multiplication:
   ordinary addition) by a weekday‑dependent weight vector
   (`weekday_weight_vector`).  Tropical addition (max) then aggregates the
   weighted spline values, yielding a *max‑plus utility* for each edge.
3. A geometric path is transformed by the lead‑lag operation (Parent B).  The
   transformed path vectors are combined with the edge‑wise utilities through
   tropical matrix‑vector multiplication, producing a single scalar that
   simultaneously encodes tree‑based cost, temporal (weekday) modulation and
   causal path information.

The three core functions below demonstrate this hybrid pipeline.
"""

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
    """Tropical addition: max(x, y). Works element‑wise on numpy arrays."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication: x + y. Works element‑wise on numpy arrays."""
    return np.add(x, y)


def t_matvec_mul(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Tropical matrix‑vector multiplication.
    (A ⊗ v)_i = max_j (A_{ij} + v_j)

    Parameters
    ----------
    A : (m, n) array
    v : (n,) array

    Returns
    -------
    result : (m,) array
    """
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.max(A + v, axis=1)


# ----------------------------------------------------------------------
# Tree utilities (from Parent A)
# ----------------------------------------------------------------------
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
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    # BFS to compute distances from root
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
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input shape: (T, d)
    Output shape: (2*T‑1, 2*d)
    """
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
    """
    Produce a probability‑like weight vector that depends on the day‑of‑week.
    `dow` follows Python's convention where Monday == 0.
    """
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
    """
    Evaluate (uniform) B‑spline basis functions of order `k` at positions `x`.

    This implementation uses the Cox‑de Boor recursion with a uniform knot
    vector that spans the supplied `grid`.  It returns a matrix `B` where
    B[i, j] = B_j(x_i) (the j‑th basis evaluated at x_i).

    Parameters
    ----------
    x : (m,) array of evaluation points
    grid : (g,) monotonic array defining the domain
    k : spline order (degree = k‑1)

    Returns
    -------
    B : (m, g‑k‑1) array
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)

    # Uniform knot vector with extra knots at the boundaries
    n_basis = len(grid) - k - 1
    if n_basis <= 0:
        raise ValueError("grid too short for the requested order")

    # Create knot vector
    knots = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))

    # Initialise zeroth‑order basis
    B = np.zeros((len(x), len(knots) - 1))
    for i in range(len(knots) - 1):
        left = knots[i]
        right = knots[i + 1]
        B[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    B[np.isclose(x, knots[-1]), -1] = 1.0  # include rightmost point

    # Recursion for higher orders
    for d in range(1, k + 1):
        B_next = np.zeros((len(x), len(knots) - d - 1))
        for i in range(len(knots) - d - 1):
            denom1 = knots[i + d] - knots[i]
            term1 = 0.0 if denom1 == 0 else ((x - knots[i]) / denom1) * B[:, i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            term2 = 0.0 if denom2 == 0 else ((knots[i + d + 1] - x) / denom2) * B[:, i + 1]
            B_next[:, i] = term1 + term2
        B = B_next
    # Trim to the number of usable basis functions
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
    """
    Compute a tropical‑max utility for each edge.

    Steps
    -----
    1. Evaluate B‑spline basis on the raw edge lengths.
    2. Weight the basis columns by the weekday weight vector.
    3. Apply tropical multiplication (addition) and then tropical addition (max)
       across the weighted spline values to obtain a scalar per edge.

    Returns
    -------
    utilities : (m,) array, one utility per edge length.
    """
    # 1. B‑spline evaluation
    B = bspline_basis(edge_lengths, spline_grid, k=spline_order)  # (m, nbasis)

    # 2. Weekday weighting
    w = weekday_weight_vector(groups, dow)                     # (nbasis,)

    if B.shape[1] != len(w):
        # If the number of spline basis functions differs from groups, truncate/pad.
        min_len = min(B.shape[1], len(w))
        B = B[:, :min_len]
        w = w[:min_len]

    # 3. Tropical operations
    weighted = t_mul(B, w)          # element‑wise addition
    utilities = np.max(weighted, axis=1)  # tropical addition across basis
    return utilities


def hybrid_tree_path_utility(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    path: Sequence[Sequence[float]],
    groups: Sequence[str],
    dow: int,
) -> float:
    """
    Full hybrid pipeline:

    * Compute tree distances (Parent A).
    * Convert each root‑to‑node distance into an edge‑wise utility using
      `fused_edge_utility`.
    * Transform the external geometric path with `lead_lag_transform`.
    * Treat the transformed path as a tropical vector and combine it with the
      edge utilities via tropical matrix‑vector multiplication.

    The result is a single scalar that reflects:
    – tree‑based cost,
    – weekday‑modulated spline shaping,
    – causal information from the path.
    """
    # Tree metrics
    _, edge_len_dict, dist_dict = tree_metrics(nodes, edges, root)

    # Gather edge lengths in a deterministic order
    edge_lengths = np.array([edge_len_dict[(u, v)] for u, v in edges], dtype=float)

    # Build a uniform grid for the spline based on observed lengths
    grid_min, grid_max = edge_lengths.min(), edge_lengths.max()
    spline_grid = np.linspace(grid_min, grid_max, num= max(5, len(edge_lengths) * 2))

    # Edge utilities (tropical scalars)
    edge_utils = fused_edge_utility(edge_lengths, groups, dow, spline_grid)

    # Assemble a tropical matrix: each row corresponds to an edge, each column to a
    # dimension of the lead‑lag transformed path.  We broadcast the edge utility
    # across columns so that tropical multiplication adds the utility to each path
    # component.
    lead_lag = lead_lag_transform(path)                     # (p, 2d)
    tropical_matrix = np.tile(edge_utils[:, np.newaxis], (1, lead_lag.shape[1]))

    # Tropical matrix‑vector product
    result_vec = t_matvec_mul(tropical_matrix, lead_lag.mean(axis=0))

    # Aggregate final scalar (tropical addition)
    final_score = np.max(result_vec)
    return float(final_score)


def fused_allocate_hybrid(
    groups: Sequence[str],
    dow: int,
    budget: float,
    edge_lengths: np.ndarray,
    spline_grid: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Demonstrates a resource‑allocation style hybrid operation.

    * Compute utilities for each edge.
    * Allocate a portion of the total `budget` proportionally to the tropical
      utilities (using softmax‑like scaling).
    * Return the allocation vector and the leftover budget.
    """
    utilities = fused_edge_utility(edge_lengths, groups, dow, spline_grid)
    # Tropical softmax: exponentiate after shifting by max to avoid overflow,
    # then normalise.
    shifted = utilities - np.max(utilities)
    exp_vals = np.exp(shifted)
    fractions = exp_vals / exp_vals.sum()
    allocation = budget * fractions
    leftover = budget - allocation.sum()
    return allocation, float(leftover)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    root = "A"

    # Dummy path (square)
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    groups = ["g0", "g1", "g2", "g3"]
    dow = 2  # Wednesday

    score = hybrid_tree_path_utility(nodes, edges, root, path, groups, dow)
    print(f"Hybrid tree‑path utility score: {score:.4f}")

    # Demonstrate allocation helper
    # Re‑use edge lengths from the tree
    _, edge_len_dict, _ = tree_metrics(nodes, edges, root)
    edge_lengths = np.array([edge_len_dict[e] for e in edges], dtype=float)
    grid = np.linspace(edge_lengths.min(), edge_lengths.max(), 10)
    alloc, left = fused_allocate_hybrid(groups, dow, budget=100.0,
                                        edge_lengths=edge_lengths,
                                        spline_grid=grid)
    print(f"Allocation per edge: {alloc}")
    print(f"Leftover budget: {left:.2f}")