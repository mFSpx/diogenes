# DARWIN HAMMER — match 4472, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_h_hybrid_hybrid_path_s_m1310_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_hybrid_ternar_m1062_s0.py (gen4)
# born: 2026-05-29T23:56:06Z

"""Hybrid NLMS‑B‑spline‑LeadLag‑TreeCost algorithm.

Parents:
- hybrid_hybrid_hybrid_nlms_h_hybrid_hybrid_path_s_m1310_s0.py (adaptive NLMS filter with B‑spline expansion,
  entropy‑modulated learning rate and circuit‑breaker safety).
- hybrid_hybrid_hybrid_label__hybrid_hybrid_ternar_m1062_s0.py (lead‑lag path transform and tree‑cost routing).

Mathematical bridge:
The lead‑lag transformed path `L(t)` is projected onto a B‑spline basis `B(t)` producing a design
matrix `X = B(L)`.  The NLMS weight update is applied to the spline coefficients `w`.  The learning
rate `μ` is jointly modulated by (i) the Shannon entropy of the raw path (as in parent A) and (ii)
the tree‑cost of a geometric graph derived from the same path (as in parent B).  A circuit‑breaker
monitors the instantaneous prediction error and forces a reset when a failure streak exceeds a
threshold, preserving the safety semantics of parent A.

The core update for a single time‑step `k` is

    e_k   = y_k - X_k·w_k
    μ_k   = μ0 / (1 + α·C_tree + β·H_path)
    w_{k+1}= w_k + (μ_k / (‖X_k‖²+ε))·e_k·X_k

where `C_tree` is the tree‑cost, `H_path` the path entropy, and `α,β` are scaling constants.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ----------------------------------------------------------------------
# Data structures (re‑used from parents)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Lead‑lag transform (parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Return the lead‑lag representation of a 2‑D (or higher‑dim) path."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T):
        if t == 0:
            out[t] = np.concatenate((path[t], np.zeros(d)))
        elif t == T - 1:
            out[2 * t - 1] = np.concatenate((np.zeros(d), path[t]))
        else:
            out[2 * t - 1] = np.concatenate((path[t - 1], path[t]))
    return out

# ----------------------------------------------------------------------
# Simple B‑spline basis matrix (degree 3, uniform knots)
# ----------------------------------------------------------------------
def bspline_design_matrix(t_vals: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """Construct a B‑spline design matrix B(t) of shape (len(t_vals), n_basis).

    Uniform knots are placed on the interval [t_min, t_max] with clamped end conditions.
    """
    t_min, t_max = t_vals.min(), t_vals.max()
    # number of internal knots
    n_internal = n_basis - (degree + 1)
    if n_internal < 0:
        raise ValueError("Not enough basis functions for the chosen degree.")
    # clamped knot vector
    knots = np.concatenate((
        np.full(degree, t_min),
        np.linspace(t_min, t_max, n_internal + 2),
        np.full(degree, t_max)
    ))
    # Cox–de Boor recursion (vectorised for each t)
    def basis(i, k, t):
        if k == 0:
            return np.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
        left = (t - knots[i]) / (knots[i + k] - knots[i] + 1e-12) * basis(i, k - 1, t)
        right = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1] + 1e-12) * basis(i + 1, k - 1, t)
        return left + right

    B = np.empty((len(t_vals), n_basis), dtype=float)
    for i in range(n_basis):
        B[:, i] = basis(i, degree, t_vals)
    return B

# ----------------------------------------------------------------------
# Entropy of a raw path (parent A)
# ----------------------------------------------------------------------
def path_entropy(path: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of the marginal distribution of path coordinates."""
    hist, _ = np.histogram(path.ravel(), bins=bins, density=True)
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

# ----------------------------------------------------------------------
# Tree cost (parent B)
# ----------------------------------------------------------------------
def tree_cost(nodes: Dict[str, Point], edges: List[Edge]) -> float:
    """Total Euclidean length of all edges (material) for an undirected tree."""
    material = 0.0
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        material += math.hypot(xa - xb, ya - yb)
    return material

# ----------------------------------------------------------------------
# Circuit breaker (parent A)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure counter that opens after `failure_threshold` consecutive errors."""
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

# ----------------------------------------------------------------------
# NLMS core update (parent A)
# ----------------------------------------------------------------------
def nlms_step(
    w: np.ndarray,
    X: np.ndarray,
    y: float,
    mu: float,
    eps: float = 1e-8
) -> Tuple[np.ndarray, float]:
    """Perform a single NLMS weight update.

    Returns the updated weight vector and the prediction error.
    """
    y_hat = X @ w
    e = y - y_hat
    norm_sq = X @ X + eps
    w_new = w + (mu / norm_sq) * e * X
    return w_new, e

# ----------------------------------------------------------------------
# Hybrid operation ----------------------------------------------------
# ----------------------------------------------------------------------
def hybrid_nlms_update(
    path: np.ndarray,
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    w: np.ndarray,
    mu0: float = 0.5,
    alpha: float = 0.01,
    beta: float = 0.1,
    eps: float = 1e-8,
    breaker: EndpointCircuitBreaker | None = None
) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid NLMS update using lead‑lag transformed path,
    B‑spline projection, entropy‑ and tree‑cost‑modulated learning rate.

    Parameters
    ----------
    path : np.ndarray
        Original (T, d) trajectory.
    nodes, edges, root : graph describing a geometric tree built from the same path.
    w : np.ndarray
        Current spline coefficient vector (size = n_basis).
    mu0, alpha, beta : scalars controlling base learning rate and modulation strength.
    eps : regularisation term for NLMS denominator.
    breaker : optional EndpointCircuitBreaker instance.

    Returns
    -------
    w_new : np.ndarray
        Updated weight vector.
    e : float
        Prediction error after the update.
    """
    # 1. Lead‑lag transform
    L = lead_lag_transform(path)                     # shape (2T‑1, 2d)

    # 2. Build a scalar parameter t for spline basis (use norm of each row)
    t_vals = np.linalg.norm(L, axis=1)

    # 3. B‑spline design matrix
    n_basis = w.shape[0]
    B = bspline_design_matrix(t_vals, n_basis)      # shape (2T‑1, n_basis)

    # 4. Target signal – for demonstration we use the sum of original coordinates
    y_target = float(path.sum())

    # 5. Modulate learning rate
    H = path_entropy(path)
    C = tree_cost(nodes, edges)
    mu = mu0 / (1.0 + alpha * C + beta * H)

    # 6. NLMS step on the aggregated design matrix (use mean over time as a single observation)
    X = B.mean(axis=0)                               # shape (n_basis,)
    w_new, e = nlms_step(w, X, y_target, mu, eps)

    # 7. Circuit‑breaker handling
    if breaker is not None:
        if abs(e) > 1.0:  # arbitrary error magnitude threshold
            breaker.record_failure()
            if breaker.open:
                # Reset weights to zero on open
                w_new = np.zeros_like(w_new)
        else:
            breaker.record_success()

    return w_new, e

# ----------------------------------------------------------------------
# Helper to generate a synthetic graph from a path (used in tests)
# ----------------------------------------------------------------------
def path_to_graph(path: np.ndarray) -> Tuple[Dict[str, Point], List[Edge], str]:
    """Create a simple chain graph where each point becomes a node."""
    nodes = {f"v{i}": tuple(coord) for i, coord in enumerate(path)}
    edges = [(f"v{i}", f"v{i+1}") for i in range(len(path) - 1)]
    root = "v0"
    return nodes, edges, root

# ----------------------------------------------------------------------
# Smoke test -----------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic 2‑D path
    T = 30
    path = np.cumsum(np.random.randn(T, 2), axis=0)

    # Graph derived from the path
    nodes, edges, root = path_to_graph(path)

    # Initialise spline coefficients (10 basis functions)
    n_basis = 10
    w = np.zeros(n_basis)

    # Circuit breaker instance
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Perform a few hybrid updates
    for epoch in range(5):
        w, err = hybrid_nlms_update(
            path=path,
            nodes=nodes,
            edges=edges,
            root=root,
            w=w,
            mu0=0.6,
            alpha=0.02,
            beta=0.15,
            breaker=breaker
        )
        print(f"Epoch {epoch+1:02d} | error={err:.4f} | breaker_open={breaker.open}")

    print("Final weights:", w)