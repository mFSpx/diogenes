# DARWIN HAMMER — match 2258, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s1.py (gen4)
# parent_b: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s2.py (gen3)
# born: 2026-05-29T23:41:30Z

"""
This module fuses the lead-lag transform and B-spline basis from hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s1.py 
and the hybrid bandit router with fold-change detection from hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s2.py.
The mathematical bridge between the two structures lies in the use of log-count ratio and 
the B-spline basis to approximate the effective number of activation patterns that influences 
the store factor in the hybrid bandit router.

The fusion of the two modules is achieved by using the B-spline basis to weight the 
log-count ratio and feed the free-energy asymptotic and the RLCT regression.
"""

import numpy as np
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Reset the bandit policy."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Compute the reward for an action based on the bandit policy."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Count the number of times an action has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

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
                if denom_l > 1e-12 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 1e-12 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the store factor based on the log-count ratio and B-spline basis."""
    grid = np.linspace(0, 1, 10)
    x = np.array([log_count_ratio])
    B = bspline_basis(x, grid)
    store_factor = B[0, 0] * count
    return store_factor

def hybrid_select_action(action_id: str, propensity: float, expected_reward: float, 
                         confidence_bound: float, algorithm: str) -> BanditAction:
    """Select an action based on the hybrid bandit router with the influence of 
    the store factor and the log-count ratio."""
    count = _count(action_id)
    log_count_ratio = math.log(count + 1)
    store_factor = _hybrid_store_factor(action_id, count, log_count_ratio)
    return BanditAction(action_id, propensity * store_factor, expected_reward, 
                        confidence_bound, algorithm)

def hybrid_rlct_estimate(sketch: np.ndarray, grid: np.ndarray) -> float:
    """Derive an RLCT estimate from the sketch-based loss curve and evaluate 
    the asymptotic free energy."""
    B = bspline_basis(sketch, grid)
    rlct_estimate = np.sum(B * sketch)
    return rlct_estimate

def fold_change_detection_series(series: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Apply the fold-change detection to a series of inputs."""
    ratios = series / np.maximum(np.abs(series), eps)
    return np.where(np.abs(ratios) > 1, ratios, 0)

if __name__ == "__main__":
    path = np.random.rand(10, 2)
    transformed_path = lead_lag_transform(path)
    print(transformed_path.shape)

    grid = np.linspace(0, 1, 10)
    x = np.array([0.5])
    B = bspline_basis(x, grid)
    print(B.shape)

    action_id = "action_1"
    count = 10.0
    log_count_ratio = math.log(count + 1)
    store_factor = _hybrid_store_factor(action_id, count, log_count_ratio)
    print(store_factor)

    sketch = np.random.rand(10)
    rlct_estimate = hybrid_rlct_estimate(sketch, grid)
    print(rlct_estimate)

    series = np.random.rand(10)
    ratios = fold_change_detection_series(series)
    print(ratios)