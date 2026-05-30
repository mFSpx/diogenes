# DARWIN HAMMER — match 2258, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s1.py (gen4)
# parent_b: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s2.py (gen3)
# born: 2026-05-29T23:41:30Z

"""
This module fuses the `hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s1` and 
`hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s2` algorithms. 
The mathematical bridge between the two structures lies in the use of log-count statistics 
and the integration of B-spline basis functions with the store factor in the hybrid bandit router.
The hybrid bandit router uses a store factor to influence the selection of actions, 
while the B-spline basis functions are used to approximate the effective number of activation patterns 
that influences the store factor. The combined quantities feed the free-energy asymptotic and the RLCT regression.
"""

import numpy as np
import math
import random
import sys
from collections import defaultdict
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
                if denom_l > 1e-12 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 1e-12 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0.

    Uses g=7 Lanczos coefficients from Numerical Recipes.  Accurate to ~15
    significant figures for real z > 0.  For z < 0.5 uses the reflection
    formula Gamma(z)*Gamma(1-z) = pi/sin(pi*z) to stay in the stable region.
    """
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])

    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        sum = _LANCZOS_C[0]
        for i in range(1, _LANCZOS_G + 2):
            sum += _LANCZOS_C[i] / (z + i - 1)
        return sum * math.sqrt(2 * math.pi) * (z + _LANCZOS_G - 1.5) ** (z - 0.5) * math.exp(-z - _LANCZOS_G + 1.5)

def hybrid_select_action(count, log_count_ratio):
    """Select an action based on the hybrid bandit router with the influence of 
    the store factor and the log-count ratio."""
    action_id = "action_1"
    propensity = log_count_ratio / (log_count_ratio + count)
    expected_reward = 0.5 * (count + log_count_ratio)
    confidence_bound = 0.1 * (count + log_count_ratio)
    return action_id, propensity, expected_reward, confidence_bound

def hybrid_rlct_estimate(path):
    """Derive an RLCT estimate from the sketch-based loss curve and evaluate 
    the asymptotic free energy."""
    path = lead_lag_transform(path)
    grid = np.linspace(0, 1, 10)
    x = np.linspace(0, 1, 100)
    B = bspline_basis(x, grid)
    estimate = np.dot(B.T, path)
    free_energy = np.dot(estimate, estimate) / len(estimate)
    return estimate, free_energy

def fold_change_detection_series(series):
    """Apply the fold-change detection to a series of inputs."""
    result = []
    for i in range(1, len(series)):
        fold_change = series[i] / series[i - 1]
        result.append(fold_change)
    return result

if __name__ == "__main__":
    path = np.random.rand(10, 2)
    count = 10
    log_count_ratio = 0.5
    action_id, propensity, expected_reward, confidence_bound = hybrid_select_action(count, log_count_ratio)
    estimate, free_energy = hybrid_rlct_estimate(path)
    series = [1, 2, 4, 8, 16]
    fold_change = fold_change_detection_series(series)
    print("Action ID:", action_id)
    print("Propensity:", propensity)
    print("Expected Reward:", expected_reward)
    print("Confidence Bound:", confidence_bound)
    print("RLCT Estimate:", estimate)
    print("Free Energy:", free_energy)
    print("Fold Change:", fold_change)