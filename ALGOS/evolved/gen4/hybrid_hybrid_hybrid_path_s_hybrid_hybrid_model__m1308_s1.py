# DARWIN HAMMER — match 1308, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s1.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_fold_change_detectio_m32_s1.py (gen2)
# born: 2026-05-29T23:35:05Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations 
of the path signature and Kolmogorov-Arnold Networks (KAN) algorithms with the 
hybrid fold-change linear trainer. The mathematical bridge between these structures 
lies in the representation of the path signature as a sequence of iterated integrals, 
which can be approximated using the B-spline basis functions employed in KANs, and 
the gain produced by the fold-change detector, which is used to modulate the effective 
learning rate of the matrix update.

The hybrid algorithm integrates the B-spline basis functions from the path signature 
and KAN algorithms with the fold-change detection update equations from the hybrid 
fold-change linear trainer. The B-spline basis functions are used to approximate the 
iterated integrals in the path signature, while the fold-change detection update 
equations are used to update the system state and weight matrix.
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

def fold_change_detection(x, threshold=0.5):
    """
    Fold-change detection algorithm.
    """
    return np.where(np.abs(x) > threshold, np.sign(x), 0)

def hybrid_update(path, W, learning_rate=0.1):
    """
    Hybrid update function that integrates the B-spline basis functions with 
    the fold-change detection update equations.
    """
    T, d = path.shape
    lead_lag_path = lead_lag_transform(path)
    B = bspline_basis(lead_lag_path[:, 0], np.linspace(0, 1, 10))
    gain = fold_change_detection(np.dot(B, W))
    W_update = W + learning_rate * np.dot(B.T, gain)
    return W_update

def hybrid_operation(path, W):
    """
    Demonstrate the hybrid operation by updating the weight matrix W using 
    the hybrid update function and then evaluating the B-spline basis functions 
    at the updated lead-lag transformed path.
    """
    W_update = hybrid_update(path, W)
    lead_lag_path = lead_lag_transform(path)
    B = bspline_basis(lead_lag_path[:, 0], np.linspace(0, 1, 10))
    return np.dot(B, W_update)

if __name__ == "__main__":
    np.random.seed(0)
    path = np.random.rand(10, 3)
    W = np.random.rand(10)
    result = hybrid_operation(path, W)
    print(result)