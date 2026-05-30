# DARWIN HAMMER — match 5770, survivor 2
# gen: 4
# parent_a: hybrid_path_signature_kan_m30_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# born: 2026-05-30T00:04:33Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations 
of the path signature and Kolmogorov-Arnold Networks (KAN) algorithms with 
the Hybrid Fractional-Memory Allocation Module.

The mathematical bridge between these structures lies in the use of the 
B-spline basis functions employed in KANs to approximate the path signature 
as a sequence of iterated integrals, and the Caputo fractional derivative 
to introduce a memory term into the allocation process.

By integrating the KAN's B-spline basis into the path signature computation 
and the fractional-memory kernel into the allocation process, we can leverage 
the expressive power of neural networks and the power-law memory kernel to 
improve the accuracy of the path signature representation and the allocation process.
"""

import numpy as np
import math
import random
import sys
from datetime import date
from pathlib import Path

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

def caputo_derivative(f, t, alpha):
    """
    Compute the Caputo fractional derivative of order alpha.
    """
    return np.array([f(t_i) / math.gamma(1 - alpha) / (t_i ** alpha) for t_i in t])

def hybrid_allocation(path, grid, k=3, alpha=0.5):
    """
    Compute the hybrid allocation using the B-spline basis and Caputo fractional derivative.
    """
    lead_lag_path = lead_lag_transform(path)
    bbasis = bspline_basis(lead_lag_path[:, 0], grid, k)
    caputo_term = caputo_derivative(lambda t: np.sum(bbasis[:, t]), np.arange(len(bbasis)), alpha)
    return np.dot(bbasis.T, caputo_term)

def summarize_hybrid_savings(path, grid, k=3, alpha=0.5):
    """
    Aggregate baseline vs. hybrid allocations and report a savings percentage.
    """
    baseline_allocation = np.sum(path, axis=0)
    hybrid_allocation = hybrid_allocation(path, grid, k, alpha)
    savings_percentage = (baseline_allocation - hybrid_allocation) / baseline_allocation * 100
    return savings_percentage

if __name__ == "__main__":
    path = np.random.rand(10, 3)
    grid = np.linspace(0, 1, 10)
    k = 3
    alpha = 0.5
    hybrid_alloc = hybrid_allocation(path, grid, k, alpha)
    print(hybrid_alloc)