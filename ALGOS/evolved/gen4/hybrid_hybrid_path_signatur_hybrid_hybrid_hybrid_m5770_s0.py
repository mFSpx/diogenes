# DARWIN HAMMER — match 5770, survivor 0
# gen: 4
# parent_a: hybrid_path_signature_kan_m30_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# born: 2026-05-30T00:04:33Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations of the path signature and Hybrid Fractional-Memory Allocation algorithms.
The mathematical bridge between these two structures lies in the use of the Caputo fractional derivative to introduce a memory term into the path signature computation.
The fractional-memory kernel is used to weight the historical iterated integrals, which are then used to modulate the path signature representation.
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

def caputo_fractional_derivative(x, alpha):
    """
    Caputo fractional derivative of order alpha.
    """
    return np.sum(np.power(x, alpha) * np.diff(x, alpha) / np.math.factorial(alpha))

def hybrid_path_signature_kan_m30_caputo(x, alpha, grid, k=3):
    """
    Hybrid path signature computation with Caputo fractional derivative.
    """
    lead_lag_path = lead_lag_transform(x)
    B = bspline_basis(lead_lag_path, grid, k=k)
    fractional_derivative = caputo_fractional_derivative(lead_lag_path, alpha)
    return B * fractional_derivative

def hybrid_fm_allocate_by_dates(dates, groups, alpha, grid, k=3):
    """
    Compute per-day, per-group allocations using the fractional-memory modulated LLM share.
    """
    T, d = dates.shape
    out = np.empty((T, len(groups)), dtype=float)
    for t in range(T):
        path_signature = hybrid_path_signature_kan_m30_caputo(dates[t], alpha, grid, k=k)
        out[t] = np.sum(path_signature, axis=1) / np.sum(path_signature)
    return out

def summarize_hybrid_fm_savings(path_signature, fm_allocation):
    """
    Aggregate baseline vs. fractional-memory modulated allocations and report a savings percentage.
    """
    savings = np.sum(path_signature * fm_allocation) / np.sum(path_signature)
    return 100 * (1 - savings)

if __name__ == "__main__":
    # Smoke test
    dates = np.random.rand(10, 5)
    groups = np.random.choice(["codex", "groq", "cohere", "local_models"], 5)
    alpha = 0.5
    grid = np.linspace(0, 1, 10)
    path_signature = hybrid_path_signature_kan_m30_caputo(dates, alpha, grid)
    fm_allocation = hybrid_fm_allocate_by_dates(dates, groups, alpha, grid)
    savings = summarize_hybrid_fm_savings(path_signature, fm_allocation)
    print(f"Savings percentage: {savings:.6f}%")