# DARWIN HAMMER — match 572, survivor 0
# gen: 5
# parent_a: hybrid_path_signature_kan_m30_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s3.py (gen4)
# born: 2026-05-29T23:29:47Z

"""
This module fuses the mathematical structures of two parent algorithms:
1. hybrid_path_signature_kan_m30_s3.py - 
   a combination of path signature and KAN machinery.
2. hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s3.py - 
   a fusion of regret-weighted ternary-decision analyzer and audit-signature pruning.

The mathematical bridge between the two parents lies in the application of 
regret-weighted probabilities to the pruning schedule of the audit-signature 
pruning algorithm, and the integration of path signature and KAN machinery 
with the regret-weighted decision-making process.

This hybrid algorithm integrates the governing equations of both parents, 
enabling a more comprehensive analysis of decision-making processes.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import timezone

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
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

def calculate_regret_weighted_probabilities(actions: list) -> np.ndarray:
    probabilities = np.array([action.expected_value for action in actions])
    regret_weights = np.array([action.risk for action in actions])
    return probabilities * regret_weights

def integrate_path_signature_with_regret(actions: list, path: np.ndarray) -> np.ndarray:
    regret_weights = calculate_regret_weighted_probabilities(actions)
    transformed_path = lead_lag_transform(path)
    signature = signature_level1(transformed_path)
    return regret_weights * signature

def hybrid_kan_layer(
    x: np.ndarray,
    spline_weights: np.ndarray,
    grid: np.ndarray,
    k: int = 3,
    actions: list = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    spline_weights = np.asarray(spline_weights, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    batch, n_in = x.shape
    n_out, n_in_w, n_basis = spline_weights.shape
    assert n_in == n_in_w, f"n_in mismatch: x has {n_in}, weights expect {n_in_w}"
    expected_n_basis = len(grid) + k - 2

    if actions is not None:
        regret_weights = calculate_regret_weighted_probabilities(actions)
        x = x * regret_weights[:, None]

    B = bspline_basis(x, grid, k)
    out = np.einsum("bi, bij -> bj", x, B)
    return out

if __name__ == "__main__":
    actions = [MathAction("action1", 0.5), MathAction("action2", 0.3)]
    path = np.random.rand(10, 2)
    print(integrate_path_signature_with_regret(actions, path))
    x = np.random.rand(10, 2)
    spline_weights = np.random.rand(2, 2, 10)
    grid = np.linspace(0, 1, 10)
    print(hybrid_kan_layer(x, spline_weights, grid, actions=actions))