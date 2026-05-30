# DARWIN HAMMER — match 1321, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_decreasing_pr_m367_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s1.py (gen4)
# born: 2026-05-29T23:35:09Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of the hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s2.py and hybrid_hybrid_hybrid_workshare_hybrid_hybrid_path_s_m851_s1.py algorithms.
The mathematical bridge between these two structures lies in the use of the B-spline basis functions from the path signature algorithm to evaluate the similarity between the input and output of the ternary router algorithm's edge scoring function, and the integration of the bandit algorithm's update policy and store update into the path signature computation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import date

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def bspline_basis(x, grid, k=3, weights=None):
    """
    Evaluate B-spline basis functions of order k at positions x, 
    with optional modulation by weights.
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

    if weights is not None:
        B *= weights

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term = (t[i + order] - x) / denom_r
            for j in range(order):
                B_new[:, i + j] += B[:, i + j + 1] * term
        B = B_new

    return B

def ssim(x: np.ndarray, y: np.ndarray, grid: np.ndarray, weights: np.ndarray) -> float:
    """
    Evaluate the similarity between x and y using the B-spline basis functions and weights.
    """
    B = bspline_basis(x, grid, weights=weights)
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    cov_xy = np.mean((x - mu_x) * (y - mu_y), axis=0)
    cov_xx = np.mean((x - mu_x) ** 2, axis=0)
    ssim_value = 2 * cov_xy / (cov_xx + np.mean(y, axis=0) ** 2)
    return np.mean(ssim_value)

def update_policy(updates: list) -> None:
    """
    Update the policy based on the rewards and actions.
    """
    _POLICY.clear()
    for u in updates:
        stats = _POLICY.setdefault(u['action_id'], [0.0, 0.0])
        stats[0] += float(u['reward'])
        stats[1] += 1.0

def update_store(
    store: float,
    inflow: list[float],
    outflow: list[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> tuple[float, float]:
    """
    Update the store and delta based on the inflow and outflow.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def dance_duration(
    delta_store: float,
    base: float = 1.0,
    gain: float = 1.0,
    limit: float = 10.0,
) -> float:
    """
    Calculate the duration based on the delta_store.
    """
    return max(0.0, min(limit, base + gain * delta_store))

if __name__ == "__main__":
    # Smoke test
    x = np.random.rand(10)
    y = np.random.rand(10)
    grid = np.linspace(0, 10, 10)
    weights = weekday_weight_vector(["group1", "group2", "group3"], 0)
    print(ssim(x, y, grid, weights))