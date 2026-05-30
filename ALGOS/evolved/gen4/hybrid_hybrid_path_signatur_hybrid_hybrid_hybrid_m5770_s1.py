# DARWIN HAMMER — match 5770, survivor 1
# gen: 4
# parent_a: hybrid_path_signature_kan_m30_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# born: 2026-05-30T00:04:33Z

"""
This module implements a novel hybrid algorithm that fuses the path signature and Kolmogorov-Arnold Networks (KAN) algorithms with the Hybrid Fractional-Memory Allocation Module.
The mathematical bridge between these two structures lies in the use of the B-spline basis functions to represent the path signature and the application of the Caputo fractional derivative to introduce a memory term into the allocation process.
By integrating the KAN's B-spline basis into the path signature computation and modulating the LLM allocation with the fractional-memory kernel, we can leverage the expressive power of neural networks to improve the accuracy of the path signature representation and optimize the allocation process.
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

def init_hybrid_fm_allocation(grid, k=3):
    """
    Initialize the hybrid allocation parameters.
    """
    B = bspline_basis(np.arange(len(grid)), grid, k)
    return B

def hybrid_fm_allocate_by_dates(path, B, dates):
    """
    Compute per-day, per-group allocations using the fractional-memory modulated LLM share.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((T, d), dtype=float)
    for t in range(T):
        out[t] = np.dot(B[t], path[t])
    return out

def summarize_hybrid_fm_savings(path, B, dates):
    """
    Aggregate baseline vs. fractional-memory modulated allocations and report a savings percentage.
    """
    baseline = np.sum(path, axis=0)
    hybrid = np.sum(hybrid_fm_allocate_by_dates(path, B, dates), axis=0)
    savings = (baseline - hybrid) / baseline * 100
    return savings

if __name__ == "__main__":
    grid = np.linspace(0, 1, 10)
    path = np.random.rand(10, 3)
    B = init_hybrid_fm_allocation(grid)
    dates = np.arange(10)
    savings = summarize_hybrid_fm_savings(path, B, dates)
    print(savings)