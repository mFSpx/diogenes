# DARWIN HAMMER — match 5580, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1321_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1969_s0.py (gen4)
# born: 2026-05-30T00:02:59Z

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import math
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    C1 = (0.01 * 1.)**2
    C2 = (0.03 * 1.)**2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)

    ssim_map = ((2 * mu_x * mu_y + C1) *
                (2 * sigma_x * sigma_y + C2) *
                (2 * mu_x * mu_y + C1) *
                (2 * sigma_x * sigma_y + C2)) / (
                   (mu_x ** 2 + mu_y ** 2 + C1) *
                   (sigma_x ** 2 + sigma_y ** 2 + C2))

    return ssim_map

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, targ):
    return np.mean((np.dot(W, x) - targ) ** 2)

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass
class RBFSurrogate:
    centers: list[list[float]]
    weights: list[float]
    epsilon: float

def hybrid_hybrid_fusion(x, W, groups, dow):
    weight_vec = weekday_weight_vector(groups, dow)
    ssim_val = ssim(x, np.dot(W, x))
    B = bspline_basis(x, np.arange(len(x)))
    return np.dot(B, W * ssim_val * weight_vec)

def hybrid_rbf_update(x, W, ssim_val, groups, dow):
    weight_vec = weekday_weight_vector(groups, dow)
    return W * ssim_val * weight_vec

def hybrid_ttt_loss(W, x, targ):
    return np.mean((np.dot(W, x) - targ) ** 2)

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(10)
    W = init_ttt(10)
    groups = ['A', 'B', 'C']
    dow = 0
    print(hybrid_hybrid_fusion(x, W, groups, dow))
    print(hybrid_rbf_update(x, W, ssim(x, np.dot(W, x)), groups, dow))
    print(hybrid_ttt_loss(W, x, np.random.rand(10)))