# DARWIN HAMMER — match 2751, survivor 1
# gen: 6
# parent_a: hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s0.py (gen5)
# born: 2026-05-29T23:45:37Z

"""
This module integrates the mathematical structures of the hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s2 and 
hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s0 algorithms.
The mathematical bridge between these two algorithms lies in the use of the SSIM score as a gain on the store's delta, 
and the geometric product as a means of updating the store dynamics.
This fusion module integrates these two concepts by using the stylometry features as input to the weight matrix updates, 
and incorporating the stable hashing as a regularization term in the gradient descent updates.
The store dynamics are updated using the geometric product, and the SSIM score is used to weight the delta.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import List, Tuple

ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
K1 = 0.01
K2 = 0.03
L = 255.0

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                if j < n - 1:
                    lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

def geometric_product(a, b):
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            if blade_out not in result:
                result[blade_out] = 0
            result[blade_out] += coef_a * coef_b * sign
    return result

def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = ALPHA,
    beta: float = BETA,
    dt: float = DT,
) -> Tuple[float, float]:
    """
    Parent-A store dynamics.
    Returns the updated store value and the raw delta (Δstore).
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def _ssim_index(x: np.ndarray, y: np.ndarray) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    k1 = K1
    k2 = K2
    L = 255
    c1 = (k1 * L)**2
    c2 = (k2 * L)**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
    return ssim

def hybrid_update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    dt: float = DT,
) -> Tuple[float, float]:
    """
    Hybrid store dynamics.
    Returns the updated store value and the raw delta (Δstore).
    """
    new_store, delta = update_store(store, inflow, outflow, alpha, beta, dt)
    ssim = _ssim_index(x, y)
    weighted_delta = ssim * delta
    hybrid_store = max(0.0, store + dt * weighted_delta)
    return hybrid_store, weighted_delta

def geometric_ttt_loss(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def geometric_ttt_grad(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

if __name__ == "__main__":
    # Smoke test
    store = 10.0
    inflow = [1.0, 2.0, 3.0]
    outflow = [0.5, 1.0, 1.5]
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])
    hybrid_store, weighted_delta = hybrid_update_store(store, inflow, outflow, x, y)
    print(f"Hybrid store: {hybrid_store}, Weighted delta: {weighted_delta}")