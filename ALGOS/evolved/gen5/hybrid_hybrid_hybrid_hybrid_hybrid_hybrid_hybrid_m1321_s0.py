# DARWIN HAMMER — match 1321, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_decreasing_pr_m367_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s1.py (gen4)
# born: 2026-05-29T23:35:09Z

"""
This module fuses the hybrid_hybrid_hybrid_ternar_hybrid_decreasing_pr_m367_s0 and 
hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s1 algorithms into a single hybrid system.
The mathematical bridge between the two structures lies in the use of the ssim function from the 
ternary router algorithm to evaluate the similarity between the input and output of the 
workshare allocator's weekday weight vector computation, and the integration of the B-spline basis 
functions from the path signature algorithm into the edge scoring function of the decreasing 
pruning algorithm.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
import json
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
            term_l = (x - t[i]) / denom_l * B[:, i]
            term_r = (t[i + order] - x) / denom_r * B[:, i + 1]
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def ssim(x: np.ndarray, y: np.ndarray) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    return (2 * mu_x * mu_y + 2 * sigma_xy) / (mu_x**2 + mu_y**2 + sigma_x**2 + sigma_y**2)

def edge_scoring(x: np.ndarray, weights: np.ndarray) -> float:
    return np.dot(x, weights) * ssim(x, weights)

def hybrid_operation(x: np.ndarray, groups: list[str], dow: int, grid: np.ndarray) -> float:
    weights = weekday_weight_vector(groups, dow)
    B = bspline_basis(x, grid, weights=weights)
    return edge_scoring(x, B.mean(axis=0))

def main():
    x = np.random.rand(10)
    groups = ["A", "B", "C"]
    dow = doomsday(2024, 1, 1)
    grid = np.linspace(0, 1, 10)
    result = hybrid_operation(x, groups, dow, grid)
    print(result)

if __name__ == "__main__":
    main()