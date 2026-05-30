# DARWIN HAMMER — match 5770, survivor 3
# gen: 4
# parent_a: hybrid_path_signature_kan_m30_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# born: 2026-05-30T00:04:33Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations of the 
path signature and Kolmogorov-Arnold Networks (KAN) algorithms from `hybrid_path_signature_kan_m30_s0.py` 
with the fractional-memory allocation process from `hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py`.
The mathematical bridge between these two structures lies in the representation of the path signature 
as a sequence of iterated integrals, which can be approximated using the B-spline basis functions 
employed in KANs, and the use of the Caputo fractional derivative to introduce a memory term into the 
allocation process.
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

    return B

def lanczos_gamma(z):
    p = [0.99999999999980993, 676.5203681218851, -1259.1392160003915, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if z < 0.5:
        z = 1 - z
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    t = z + len(p) - 1.5
    return np.sqrt(2 * np.pi) * t**(z - 0.5) * np.exp(-t) * x

def caputo_derivative(x, alpha):
    """
    Compute the Caputo derivative of order alpha.
    """
    return (1 / (math.gamma(1 - alpha))) * np.sum([x[i] * (i**(-alpha - 1)) for i in range(len(x))])

def hybrid_allocate(path, alpha, grid):
    """
    Compute the hybrid allocation using the path signature and Caputo derivative.
    """
    transformed_path = lead_lag_transform(path)
    basis = bspline_basis(np.arange(len(transformed_path)), grid)
    allocation = np.zeros(len(transformed_path))
    for i in range(len(transformed_path)):
        allocation[i] = caputo_derivative(transformed_path[:i+1], alpha)
    return allocation

def hybrid_summary(path, alpha, grid):
    """
    Summarize the hybrid allocation.
    """
    allocation = hybrid_allocate(path, alpha, grid)
    return np.sum(allocation)

if __name__ == "__main__":
    path = np.random.rand(10, 2)
    alpha = 0.5
    grid = np.linspace(0, 1, 10)
    print(hybrid_summary(path, alpha, grid))