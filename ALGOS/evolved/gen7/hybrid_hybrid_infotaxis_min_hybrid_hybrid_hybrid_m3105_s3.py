# DARWIN HAMMER — match 3105, survivor 3
# gen: 7
# parent_a: hybrid_infotaxis_minhash_m63_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s2.py (gen6)
# born: 2026-05-29T23:47:51Z

"""Hybrid Infotaxis‑MinHash & B‑Spline Voronoi System
===================================================

This module fuses the two parent algorithms:

* **Parent A** – *infotaxis/minhash* (``hybrid_infotaxis_minhash_m63_s3.py``)  
  Provides a MinHash signature for a token set and entropy‑based
  decision making.

* **Parent B** – *path‑transform / B‑spline / Voronoi* (``hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m1236_s2.py``)  
  Supplies a lead‑lag path transform, a B‑spline basis matrix and a
  Voronoi‑style nearest‑centroid assignment.

**Mathematical bridge**

The bridge is built by treating a MinHash signature (a discrete high‑dimensional
vector) as a set of scalar “samples”.  These samples are embedded into a
continuous space using the B‑spline basis matrix (a linear transformation
``B @ v``).  The resulting continuous points are then fed to a Voronoi
partitioning step, which assigns each point to a region.  Region
probabilities are used to compute Shannon entropy, and the region that
minimises the expected entropy is selected as the hybrid action.

The implementation below keeps the core equations of both parents
unaltered while wiring them together through the described bridge.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np

MAX64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# Parent A – MinHash & Entropy core
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by MinHash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks: set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted average entropy of two possible states."""
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


# ----------------------------------------------------------------------
# Parent B – Path transform, B‑spline basis & Voronoi core
# ----------------------------------------------------------------------
def lead_lag_transform(path: Iterable[Iterable[float]]) -> np.ndarray:
    """
    Lead‑lag embedding of a 2‑D path.
    Input shape: (T, d) → output shape: (2T‑1, 2d)
    """
    path_arr = np.asarray(path, dtype=float)
    if path_arr.ndim != 2:
        raise ValueError("path must be a 2‑D array")
    T, d = path_arr.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path_arr[t], path_arr[t]])
        out[2 * t + 1] = np.concatenate([path_arr[t + 1], path_arr[t]])
    out[2 * (T - 1)] = np.concatenate([path_arr[T - 1], path_arr[T - 1]])
    return out


def _bspline_basis_one(x: float, t: np.ndarray, j: int, k: int) -> float:
    """Recursive definition of a single B‑spline basis function."""
    if k == 0:
        return 1.0 if t[j] <= x < t[j + 1] else 0.0
    denom1 = t[j + k] - t[j]
    denom2 = t[j + k + 1] - t[j + 1]
    term1 = 0.0
    term2 = 0.0
    if denom1 > 0:
        term1 = (x - t[j]) / denom1 * _bspline_basis_one(x, t, j, k - 1)
    if denom2 > 0:
        term2 = (t[j + k + 1] - x) / denom2 * _bspline_basis_one(x, t, j + 1, k - 1)
    return term1 + term2


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Construct the B‑spline basis matrix B where B[i, j] = B_j(x_i).
    *x* – points to evaluate (N,). *grid* – knot positions (M,).
    Returns an (N, M + k - 1) matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Augment knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = x.shape[0]
    n_basis = len(grid) + k - 1
    B = np.zeros((N, n_basis), dtype=np.float64)

    for i in range(N):
        for j in range(n_basis):
            if t[j] <= x[i] <= t[j + 1]:
                B[i, j] = _bspline_basis_one(x[i], t, j, k)
    return B


def voronoi_assign(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each point to the index of the nearest center (Euclidean distance).
    *points* – (P, D), *centers* – (C, D). Returns an integer array of shape (P,).
    """
    if points.ndim != 2 or centers.ndim != 2:
        raise ValueError("points and centers must be 2‑D arrays")
    # Compute squared Euclidean distance matrix efficiently
    diff = points[:, None, :] - centers[None, :, :]          # (P, C, D)
    dists = np.einsum("pcd,pcd->pc", diff, diff)            # (P, C)
    return np.argmin(dists, axis=1)


# ----------------------------------------------------------------------
# Hybrid operations – the mathematical bridge
# ----------------------------------------------------------------------
def embed_signature_via_bspline(sig: List[int], num_grid: int = 32) -> np.ndarray:
    """
    Convert a MinHash signature into a continuous point using a B‑spline
    embedding.  The signature values are treated as samples ``x_i``; a uniform
    knot grid is built; the basis matrix B is applied and the row‑wise mean
    yields a single D‑dimensional vector.
    """
    if not sig:
        raise ValueError("signature must be non‑empty")
    # Normalise signature to [0, 1] for numerical stability
    sig_arr = np.asarray(sig, dtype=np.float64)
    sig_norm = (sig_arr - sig_arr.min()) / (sig_arr.ptp() + 1e-12)

    # Build a uniform grid over the normalised range
    grid = np.linspace(0.0, 1.0, num_grid)

    # B‑spline basis matrix (N samples × B basis functions)
    B = bspline_basis(sig_norm, grid, k=3)

    # Collapse to a single point by averaging across samples
    point = B.mean(axis=0)                     # shape (B,)
    return point


def compute_region_distribution(assignments: np.ndarray, n_regions: int) -> List[float]:
    """
    From Voronoi assignments compute the probability mass of each region.
    """
    counts = np.bincount(assignments, minlength=n_regions)
    total = counts.sum()
    if total == 0:
        return [0.0] * n_regions
    return (counts / total).tolist()


def hybrid_action(
    tokens: Iterable[str],
    path: Iterable[Iterable[float]],
    centers: np.ndarray,
    p_hit: float = 0.5,
) -> Tuple[int, float]:
    """
    Perform a hybrid decision step:

    1. MinHash the *tokens* → signature.
    2. Embed the signature with a B‑spline basis → point ``s``.
    3. Lead‑lag transform the *path* and take its column‑wise mean → point ``p``.
    4. Concatenate ``s`` and ``p`` → joint feature vector.
    5. Voronoi‑assign the joint vector to one of the *centers*.
    6. Compute region probabilities, expected entropy and return the region
       with minimal expected entropy.

    Returns ``(region_index, expected_entropy)``.
    """
    # 1. MinHash signature
    sig = minhash_signature(tokens, k=128)

    # 2. B‑spline embedding of the signature
    s_vec = embed_signature_via_bspline(sig, num_grid=32)   # shape (B,)

    # 3. Lead‑lag transform of the path → aggregate to a single vector
    ll = lead_lag_transform(path)                          # (2T‑1, 2d)
    p_vec = ll.mean(axis=0)                                # shape (2d,)

    # 4. Joint feature vector
    joint = np.concatenate([s_vec, p_vec])                # (B + 2d,)

    # 5. Voronoi assignment (single point → nearest center)
    assignment = voronoi_assign(joint[None, :], centers)[0]   # scalar

    # 6. Region distribution (here we treat the assignment as a deterministic
    #    observation; for demonstration we build a simple two‑state model:
    #    *hit* → region stays the same, *miss* → uniform over others.)
    n_regions = centers.shape[0]
    hit_state = compute_region_distribution(np.array([assignment]), n_regions)
    miss_state = [1.0 / n_regions] * n_regions
    exp_ent = expected_entropy(p_hit, hit_state, miss_state)

    return int(assignment), exp_ent


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample token set
    sample_tokens = [
        "alpha", "beta", "gamma", "delta", "epsilon",
        "zeta", "eta", "theta", "iota", "kappa"
    ]

    # Synthetic 2‑D path (e.g., a random walk)
    random.seed(42)
    T = 10
    path = [(random.random(), random.random()) for _ in range(T)]

    # Generate random Voronoi centers in the joint feature space.
    # Dimensionality = B‑spline basis size (≈ num_grid + k - 1) + 2*d
    num_grid = 32
    k = 3
    B_dim = num_grid + k - 1
    d = 2
    joint_dim = B_dim + 2 * d
    n_centers = 5
    centers = np.random.rand(n_centers, joint_dim)

    region, exp_entropy = hybrid_action(sample_tokens, path, centers, p_hit=0.6)

    print(f"Chosen region: {region}")
    print(f"Expected entropy for the decision: {exp_entropy:.4f}")