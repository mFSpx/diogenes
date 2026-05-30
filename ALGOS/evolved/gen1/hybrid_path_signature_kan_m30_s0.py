# DARWIN HAMMER — match 30, survivor 0
# gen: 1
# parent_a: path_signature.py (gen0)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:23:27Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations of the path signature and Kolmogorov-Arnold Networks (KAN) algorithms.
The mathematical bridge between these two structures lies in the representation of the path signature as a sequence of iterated integrals, which can be approximated using the B-spline basis functions employed in KANs.
By integrating the KAN's B-spline basis into the path signature computation, we can leverage the expressive power of neural networks to improve the accuracy of the path signature representation.
"""

import numpy as np
import math
import random
import sys
import pathlib

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    assert B.shape == (N, n_basis), (
        f"basis shape mismatch: got {B.shape}, expected ({N}, {n_basis})"
    )
    return B

def hybrid_signature(path, depth=3, k=3):
    """
    Compute the hybrid signature by integrating the B-spline basis into the path signature computation.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    increments = np.diff(path, axis=0)

    grid = np.linspace(0, 1, 10)
    B = bspline_basis(np.linspace(0, 1, T), grid, k)

    S = [np.zeros(d ** k) for k in range(1, depth + 1)]

    for t in range(T - 1):
        dx = increments[t]
        b = B[t]
        for k in range(depth - 1, 0, -1):
            S[k] = S[k] + np.outer(S[k - 1].reshape(-1), dx).ravel() * b
        S[0] = S[0] + dx * b

    return [S[k].reshape((d,) * (k + 1)) for k in range(depth)]

def hybrid_lead_lag_signature(path, depth=3, k=3):
    """
    Compute the hybrid lead-lag signature by applying the lead-lag transform to the path and then computing the hybrid signature.
    """
    path = lead_lag_transform(path)
    return hybrid_signature(path, depth, k)

def hybrid_signature_flat(path, depth=3, k=3):
    """
    Compute the hybrid signature and flatten it into a 1D feature vector.
    """
    return np.concatenate([s.ravel() for s in hybrid_signature(path, depth, k)])

if __name__ == "__main__":
    T = 50
    t = np.linspace(0, 1, T)

    path_line = np.column_stack([t, t])
    path_arch = np.column_stack([t, np.sin(np.pi * t)])

    sig_line = hybrid_signature(path_line, depth=3)
    sig_arch = hybrid_signature(path_arch, depth=3)

    print("=== Hybrid Signature comparison (same endpoints, different shapes) ===")
    print(f"Level-1 line: {sig_line[0]}")
    print(f"Level-1 arch: {sig_arch[0]}")
    print(f"Level-1 equal: {np.allclose(sig_line[0], sig_arch[0])}")
    print(f"Level-2 line:\n{sig_line[1]}")
    print(f"Level-2 arch:\n{sig_arch[1]}")
    print(f"Level-2 equal: {np.allclose(sig_line[1], sig_arch[1])}")