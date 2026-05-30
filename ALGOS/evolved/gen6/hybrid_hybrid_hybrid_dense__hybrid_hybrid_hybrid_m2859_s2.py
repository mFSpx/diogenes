# DARWIN HAMMER — match 2859, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1726_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s0.py (gen4)
# born: 2026-05-29T23:46:28Z

"""
Hybrid Algorithm: Dense Associative Memory + Path Signature + Fisher Modulation

Parents:
- hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1726_s1.py (Dense Associative Memory with softmax and Fisher score)
- hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s0.py (Lead‑lag path transform and B‑spline based signature)

Mathematical Bridge:
The bridge is the common use of high‑dimensional vector representations that are
subsequently normalised/weighted by a probability‑like function.
We first map a time‑series path to a high‑dimensional signature vector via the
lead‑lag transform followed by a B‑spline basis projection (Parent B).  This
signature serves as a query vector for a Dense Associative Memory (Parent A).
Before the associative retrieval we modulate each component of the query by
a Fisher‑information‑derived weight, computed as the derivative of a Gaussian
beam (the Fisher score).  The weighted query is finally passed through the
softmax‑based retrieval rule of the Dense AM, yielding a hybrid pattern
retrieval that fuses both topologies.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import hashlib

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1‑D array."""
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _lse(z: np.ndarray) -> float:
    """Log‑sum‑exp of a 1‑D array (numerically stable)."""
    z = np.asarray(z, dtype=float)
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


def energy(xi: np.ndarray, M: np.ndarray, beta: float = 1.0) -> float:
    """Dense Associative Memory energy E(xi)."""
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * xi @ xi
    return -beta**-1 * np.log(np.sum(np.exp(beta * M @ xi))) + lse_term + quadratic_term


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """
    Fisher score (derivative of the log‑likelihood) for a Gaussian beam.
    Returns d/dθ log p(θ) ≈ (-(θ-center)/width²) * intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = -(theta - center) / (width ** 2) * intensity
    return derivative


def fisher_weighted_softmax(vec: np.ndarray,
                           centers: np.ndarray,
                           widths: np.ndarray) -> np.ndarray:
    """
    Apply Fisher‑derived weighting to a vector and then softmax‑normalise.

    Parameters
    ----------
    vec : (d,) array
        Raw query / feature vector.
    centers, widths : (d,) arrays
        Parameters of the Gaussian beams used for Fisher weighting.

    Returns
    -------
    (d,) array
        Softmax‑normalised, Fisher‑modulated vector.
    """
    vec = np.asarray(vec, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)

    fisher_weights = np.vectorize(fisher_score)(vec, centers, widths)
    weighted = vec * fisher_weights
    return _softmax(weighted)


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
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
    Returns a (len(x), n_basis) matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    B = np.zeros((N, len(t) - 1), dtype=np.float64)

    # order 1 (piecewise constant)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # higher orders
    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l * B[:, i]) if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r * B[:, i + 1]) if denom_r > 0 else np.zeros(N)

            B_new[:, i] = term_l + term_r
        B = B_new

    return B


def path_signature_approx(path: np.ndarray,
                          grid: np.ndarray,
                          spline_order: int = 3) -> np.ndarray:
    """
    Approximate a path signature by projecting the lead‑lag transformed path
    onto a B‑spline basis. The resulting vector can be used as a high‑dimensional
    representation of the path.

    Parameters
    ----------
    path : (T, d) array
        Input time‑series.
    grid : (g,) array
        Knot positions for the spline basis (typically uniform).
    spline_order : int
        Order of the B‑spline (default cubic).

    Returns
    -------
    sig : (d * 2 * g,) array
        Flattened signature vector.
    """
    ll = lead_lag_transform(path)                # (2T‑1, 2d)
    T_ll, dim_ll = ll.shape

    # Parameterise the transformed path by a scalar “time” vector.
    t_param = np.linspace(0, 1, T_ll)

    B = bspline_basis(t_param, grid, k=spline_order)   # (T_ll, n_basis)
    coeffs = B.T @ ll                                   # (n_basis, 2d)

    return coeffs.ravel()                               # flatten to 1‑D


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def dense_associative_retrieve(query: np.ndarray,
                               memory: np.ndarray,
                               beta: float = 1.0) -> np.ndarray:
    """
    Retrieve a stored pattern from `memory` using a softmax over inner products.
    This follows the classic Dense Associative Memory retrieval rule.

    Returns the weighted sum of stored patterns.
    """
    scores = beta * (memory @ query)          # (N,)
    probs = _softmax(scores)                 # (N,)
    return probs @ memory                    # (d,)


def hybrid_path_memory_query(path: np.ndarray,
                             memory: np.ndarray,
                             beta: float = 1.0,
                             grid: np.ndarray = None,
                             centers: np.ndarray = None,
                             widths: np.ndarray = None) -> np.ndarray:
    """
    Full hybrid operation:
    1. Transform the input path into a high‑dimensional signature.
    2. Modulate the signature with Fisher‑derived weights and softmax‑normalise.
    3. Use the resulting vector as a query for Dense Associative Memory retrieval.

    Parameters
    ----------
    path : (T, d) array
        Input time‑series.
    memory : (N, d_mem) array
        Stored patterns. `d_mem` must match the signature dimension.
    beta : float
        Inverse temperature for the associative retrieval.
    grid : (g,) array or None
        Knot grid for B‑spline projection. If None, a default uniform grid of size 10 is used.
    centers, widths : (d_mem,) arrays or None
        Parameters for Fisher weighting. If None, they are set to zeros and ones respectively.

    Returns
    -------
    retrieved : (d_mem,) array
        Retrieved pattern from memory.
    """
    # 1️⃣ Signature
    if grid is None:
        grid = np.linspace(0, 1, 10)
    sig = path_signature_approx(path, grid)          # (d_sig,)

    # 2️⃣ Fisher weighting + softmax
    d_sig = sig.shape[0]
    if centers is None:
        centers = np.zeros(d_sig)
    if widths is None:
        widths = np.ones(d_sig)
    weighted_query = fisher_weighted_softmax(sig, centers, widths)

    # 3️⃣ Retrieval
    if memory.shape[1] != d_sig:
        raise ValueError(f"Memory dimension {memory.shape[1]} does not match signature dimension {d_sig}")
    return dense_associative_retrieve(weighted_query, memory, beta)


def hybrid_energy_path(query_path: np.ndarray,
                       memory: np.ndarray,
                       beta: float = 1.0,
                       grid: np.ndarray = None) -> float:
    """
    Compute the Dense Associative Memory energy of a path query.
    The path is first converted to a signature (as in the hybrid pipeline)
    and then the standard energy function is applied.
    """
    if grid is None:
        grid = np.linspace(0, 1, 10)
    sig = path_signature_approx(query_path, grid)
    return energy(sig, memory, beta)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create a dummy memory matrix (10 patterns)
    d_mem = 2 * 3 * 10   # Example: path dim=3, spline order cubic, grid size=10 => signature dim
    memory = np.random.randn(10, d_mem)

    # Generate a random path (T=15, d=3)
    T, d = 15, 3
    path = np.cumsum(np.random.randn(T, d), axis=0)  # random walk

    # Run hybrid retrieval
    retrieved = hybrid_path_memory_query(
        path=path,
        memory=memory,
        beta=1.5,
        grid=np.linspace(0, 1, 10)
    )
    print("Retrieved pattern shape:", retrieved.shape)

    # Compute hybrid energy for sanity check
    eng = hybrid_energy_path(path, memory, beta=1.5, grid=np.linspace(0, 1, 10))
    print("Hybrid energy:", eng)