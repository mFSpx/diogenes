# DARWIN HAMMER — match 882, survivor 0
# gen: 5
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s3.py (gen1)
# parent_b: hybrid_sketches_hybrid_hybrid_hybrid_m302_s1.py (gen4)
# born: 2026-05-29T23:31:22Z

"""
Hybrid Algorithm: Fusing NLMS and Lead-Lag B-Spline Path Signatures

This module fuses the core topologies of two parent algorithms:
1. hybrid_nlms_omni_chaotic_sprint_m59_s3.py (NLMS)
2. hybrid_sketches_hybrid_hybrid_hybrid_m302_s1.py (Lead-Lag B-Spline Path Signatures)

The mathematical bridge between the two algorithms lies in the use of
adaptive filtering (NLMS) and banded path signatures (Lead-Lag B-Spline).
The hybrid algorithm integrates the NLMS update rule with the lead-lag
transform and B-spline basis functions to create a novel, unified system.

The interface between the two algorithms is established through the use of
a banded, lead-lag transformed input signal, which is then fed into the
NLMS update rule. The resulting hybrid system enables adaptive filtering
of banded, lead-lag transformed signals.

"""

import numpy as np
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random
import sys

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def bspline_basis(x, grid, k=3):
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
                if denom_l > 1e-12 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 1e-12 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    assert B.shape == (N, n_basis), (
        f"basis shape mismatch: got {B.shape}, expected ({N}, {n_basis})"
    )

    return B

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error

def hybrid_nlms_lead_lag(
    weights: np.ndarray,
    path: np.ndarray,
    target: float,
    grid: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
    k: int = 3,
) -> Tuple[np.ndarray, float]:
    lead_lag_path = lead_lag_transform(path)
    B = bspline_basis(lead_lag_path, grid, k=k)
    banded_path = np.sum(B * lead_lag_path, axis=1)
    new_weights, error = nlms_update(weights, banded_path, target, mu, eps)
    return new_weights, error

def generate_random_path(T: int, d: int) -> np.ndarray:
    return np.random.rand(T, d)

def generate_random_grid(num_nodes: int) -> np.ndarray:
    return np.random.rand(num_nodes)

if __name__ == "__main__":
    T, d = 10, 3
    path = generate_random_path(T, d)
    grid = generate_random_grid(10)
    weights = np.random.rand(d)
    target = 1.0

    new_weights, error = hybrid_nlms_lead_lag(weights, path, target, grid)
    print(f"New weights: {new_weights}")
    print(f"Error: {error}")