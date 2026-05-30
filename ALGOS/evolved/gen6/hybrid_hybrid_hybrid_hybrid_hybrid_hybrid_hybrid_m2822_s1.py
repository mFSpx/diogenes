# DARWIN HAMMER — match 2822, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m498_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m1687_s0.py (gen5)
# born: 2026-05-29T23:46:13Z

"""Hybrid algorithm merging:
- Parent A: lead‑lag transform & B‑spline basis (dimensionality reduction).
- Parent B: Bayesian update & Hoeffding bound applied to graph‑tree cost.

Mathematical bridge:
Both families manipulate a *probabilistic weight* on features.
The lead‑lag transformed path yields high‑dimensional feature vectors.
These vectors are smoothed by B‑splines, then re‑weighted by a Bayesian
posterior (prior × likelihood / evidence).  The Hoeffding bound provides
a statistical guarantee on the significance of each weighted feature,
which is finally injected into the graph‑tree cost as an additive penalty.
Thus the hybrid system fuses topological data analysis with probabilistic
decision‑theoretic guarantees in a single pipeline.
"""

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dim)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of order k at positions x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Knot vector with clamped ends
    t = np.concatenate([
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1]),
    ])

    n_basis = len(grid) + k - 1
    N = len(x)

    # Order‑1 (piecewise constant) basis
    B = np.zeros((N, n_basis), dtype=np.float64)
    for i in range(n_basis):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recurrence for higher orders
    for d in range(2, k + 1):
        for i in range(n_basis - d + 1):
            left_den = t[i + d - 1] - t[i]
            right_den = t[i + d] - t[i + 1]

            left = ((x - t[i]) / left_den) * B[:, i] if left_den != 0 else 0.0
            right = ((t[i + d] - x) / right_den) * B[:, i + 1] if right_den != 0 else 0.0
            B[:, i] = left + right

    # Trim the extra columns introduced by the recurrence
    B = B[:, : n_basis - (k - 1)]
    return B


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Simple material cost of a tree: sum of Euclidean edge lengths.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    # Depth‑first traversal just to verify connectivity (not used further)
    visited = {root}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in visited:
                visited.add(b)
                stack.append(b)
    return material


def bayesian_update(prior: float, likelihood: float, evidence: float) -> float:
    """Posterior = (prior × likelihood) / evidence."""
    if evidence == 0:
        raise ZeroDivisionError("evidence must be non‑zero")
    return (prior * likelihood) / evidence


def hoeffding_bound(confidence: float, num_samples: int) -> float:
    """
    Hoeffding bound for a bounded random variable in [0,1]:
        ε = sqrt( ln(1/(1‑confidence)) / (2 n) )
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return math.sqrt(math.log(1.0 / (1.0 - confidence)) / (2.0 * num_samples))


# ----------------------------------------------------------------------
# Hybrid core functions (demonstrate the fused operation)
# ----------------------------------------------------------------------
def hybrid_feature_engineering(path: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    1. Lead‑lag transform the raw path.
    2. For each transformed channel compute B‑spline basis values.
    3. Aggregate (mean) across the time dimension to obtain a compact feature matrix.
    Returns shape (num_basis, num_channels).
    """
    ll = lead_lag_transform(path)                     # (2T‑1, 2d)
    features: List[np.ndarray] = []
    for col in range(ll.shape[1]):
        col_vals = ll[:, col]
        B = bspline_basis(col_vals, grid)            # (2T‑1, n_basis)
        # Collapse time dimension by averaging basis coefficients
        features.append(B.mean(axis=0))
    return np.column_stack(features)                 # (n_basis, 2d)


def hybrid_weighted_features(
    features: np.ndarray,
    prior: float = 0.5,
    likelihood: float = 0.7,
) -> np.ndarray:
    """
    Apply a Bayesian posterior weight to every feature entry.
    The same posterior is used for all entries (scalar weighting).
    """
    evidence = prior * likelihood + (1.0 - prior) * (1.0 - likelihood)
    posterior = bayesian_update(prior, likelihood, evidence)
    return features * posterior


def hybrid_decision_mask(
    weighted_features: np.ndarray,
    confidence: float = 0.95,
) -> np.ndarray:
    """
    Determine which feature dimensions are statistically significant.
    Uses the Hoeffding bound on the empirical variance of each column.
    Returns a boolean mask of shape (num_features,).
    """
    n_samples = weighted_features.shape[0]
    bound = hoeffding_bound(confidence, n_samples)
    variances = np.var(weighted_features, axis=0)
    return variances > bound


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    weighted_features: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Base tree material cost plus a penalty proportional to the number
    of statistically significant feature dimensions (as identified by
    `hybrid_decision_mask`).  The penalty term embodies the extra
    information the hybrid representation contributes to the structure.
    """
    base = tree_cost(nodes, edges, root)
    sig_mask = hybrid_decision_mask(weighted_features, confidence)
    penalty = sig_mask.sum() * 0.1  # tunable scalar
    return base + penalty


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic path (random walk) – 5 timesteps, 2‑D
    np.random.seed(42)
    steps = np.random.randn(5, 2)
    path = np.cumsum(steps, axis=0)

    # Grid for B‑spline basis (covering observed range)
    min_val, max_val = path.min(), path.max()
    grid = np.linspace(min_val, max_val, num=6)  # 6 knots

    # Graph for tree cost
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 2.0),
        "C": (3.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C")]
    root = "A"

    # Hybrid pipeline
    feats = hybrid_feature_engineering(path, grid)
    weighted = hybrid_weighted_features(feats, prior=0.6, likelihood=0.8)
    mask = hybrid_decision_mask(weighted, confidence=0.97)
    cost = hybrid_tree_cost(nodes, edges, root, weighted, confidence=0.97)

    print("Lead‑lag + B‑spline feature shape:", feats.shape)
    print("Weighted features (sample):", weighted[:2])
    print("Significant feature mask:", mask)
    print("Hybrid tree cost:", cost)