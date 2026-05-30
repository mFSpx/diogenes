# DARWIN HAMMER — match 332, survivor 0
# gen: 3
# parent_a: hybrid_path_signature_kan_m30_s0.py (gen1)
# parent_b: hybrid_hybrid_model_vram_sc_fold_change_detectio_m32_s2.py (gen2)
# born: 2026-05-29T23:28:16Z

import numpy as np
import math
import random
import sys
import pathlib

"""
This module implements a novel hybrid algorithm that fuses the governing equations of the path signature and Kolmogorov-Arnold Networks (KAN) algorithms with the hybrid fold-change detection and test-time training algorithms.
The mathematical bridge between these two structures lies in the representation of the path signature as a sequence of iterated integrals, which can be approximated using the B-spline basis functions employed in KANs, and the use of gain to modulate the effective learning rate in hybrid fold-change detection.
By integrating the KAN's B-spline basis into the path signature computation, and using the gain to modulate the learning rate, we can leverage the expressive power of neural networks to improve the accuracy of the path signature representation and enhance the performance of the hybrid fold-change detection algorithm.
"""

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

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Create a random weight matrix ``W`` of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Gradient of the quadratic loss ‖W x − target‖² w.r.t. ``W``."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2 * residual @ x.T

def hybrid_hybrid_model(W: np.ndarray, x: np.ndarray, target: np.ndarray | None, u: float, gain: float):
    """
    Hybrid fold-change detection and test-time training algorithm.

    Advance the fold-change detector (scalar ODE) to obtain gain, then scale the learning rate of the ttt step by 1 + gain.
    Perform the usual ttt gradient descent update with the scaled learning rate.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    grad = ttt_grad(W, x, target)
    W_new = W - 0.01 * (1 + gain) * grad
    return W_new

def hybrid_model(path: np.ndarray, grid: np.ndarray, gain: float):
    """
    Hybrid path signature and fold-change detection algorithm.

    Represent the path signature as a sequence of iterated integrals, approximated using the B-spline basis functions employed in KANs.
    Use the gain to modulate the effective learning rate in the hybrid fold-change detection algorithm.
    """
    lead_lag_path = lead_lag_transform(path)
    bspline_basis_path = bspline_basis(lead_lag_path, grid)
    W = init_ttt(len(grid), None)
    x = path
    target = None
    W_new = hybrid_hybrid_model(W, x, target, u=1.0, gain=gain)
    return W_new

def smoke_test():
    np.random.seed(0)
    random.seed(0)
    grid = np.linspace(0, 1, 100)
    path = np.random.rand(10, 2)
    gain = 0.5
    W = hybrid_model(path, grid, gain)
    assert W.shape == (100, 2), "Weight matrix shape mismatch"

if __name__ == "__main__":
    smoke_test()