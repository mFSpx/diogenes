# DARWIN HAMMER — match 332, survivor 1
# gen: 3
# parent_a: hybrid_path_signature_kan_m30_s0.py (gen1)
# parent_b: hybrid_hybrid_model_vram_sc_fold_change_detectio_m32_s2.py (gen2)
# born: 2026-05-29T23:28:16Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations of 
the path signature and Kolmogorov-Arnold Networks (KAN) algorithms with the 
hybrid fold-change linear trainer. The mathematical bridge between these structures 
lies in the representation of the path signature as a sequence of iterated integrals, 
which can be approximated using the B-spline basis functions employed in KANs, and 
the gain produced by the fold-change detector, which is used to modulate the effective 
learning rate of the matrix update.
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

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """
    Create a random weight matrix W of shape (d_out, d_in).
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """
    Gradient of the quadratic loss ‖W x - target‖² w.r.t. W.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2 * residual @ x.T

def fold_change_detection(x: float, y: float, u: float) -> float:
    """
    Evolve a scalar pair (x, y) according to a feed-forward ODE that computes a gain 
    proportional to the fold-change u / |x| of an external signal u.
    """
    gain = u / abs(x) if x != 0 else 0
    return gain

def hybrid_update(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None, u: float = 0.0) -> np.ndarray:
    """
    Hybrid update step: advance the fold-change detector, scale the learning rate, 
    and perform the ttt gradient descent update.
    """
    gain = fold_change_detection(x[0], x[1], u)
    scaled_grad = ttt_grad(W, x, target) * (1 + gain)
    return W - 0.01 * scaled_grad

def hybrid_path_signature(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute the path signature using the B-spline basis and the hybrid update step.
    """
    B = bspline_basis(x, grid, k)
    W = init_ttt(len(x), len(grid))
    for i in range(len(x) - 1):
        W = hybrid_update(W, x[i], x[i + 1], u=0.1)
    return W @ B.T

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    grid = np.array([0, 1, 2, 3, 4])
    result = hybrid_path_signature(x, grid)
    print(result)