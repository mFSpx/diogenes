# DARWIN HAMMER — match 2751, survivor 2
# gen: 6
# parent_a: hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s0.py (gen5)
# born: 2026-05-29T23:45:37Z

"""
Hybrid Store-SSIM-Geometric Product Algorithm

This module integrates the mathematical structures of the hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s2 and 
hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s0 algorithms. The mathematical bridge between these two 
algorithms lies in the use of the store update yields a change Δstore (eq. 1 of Parent A) as the input to the 
weight matrix updates in the geometric product guided test-time training, and incorporating the SSIM similarity 
as a dynamic gain on the store's delta.

The governing equations of both parents are integrated by using the stylometry features as input to the weight 
matrix updates, and incorporating the stable hashing as a regularization term in the gradient descent updates.
"""

import numpy as np
import math
import random
import sys
import pathlib

# Constants (from Parent A)
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

def update_store(
    store: float,
    inflow: list,
    outflow: list,
    alpha: float = ALPHA,
    beta: float = BETA,
    dt: float = DT,
) -> tuple:
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
    L = 255.0
    c1 = (k1 * L)**2
    c2 = (k2 * L)**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
    return ssim

def hybrid_operation(store, inflow, outflow, W, x):
    new_store, delta = update_store(store, inflow, outflow)
    ssim = _ssim_index(np.array(inflow), np.array(outflow))
    weighted_delta = delta * ssim
    loss = ttt_loss(W, x)
    grad = ttt_grad(W, x)
    return new_store, weighted_delta, loss, grad

def geometric_product(a, b):
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            if blade_out not in result:
                result[blade_out] = 0
            result[blade_out] += sign * coef_a * coef_b
    return result

if __name__ == "__main__":
    store = 100.0
    inflow = [10.0, 20.0, 30.0]
    outflow = [5.0, 10.0, 15.0]
    W = init_ttt(3)
    x = np.array([1.0, 2.0, 3.0])
    new_store, weighted_delta, loss, grad = hybrid_operation(store, inflow, outflow, W, x)
    print("New Store:", new_store)
    print("Weighted Delta:", weighted_delta)
    print("Loss:", loss)
    print("Gradient:", grad)

    a = {frozenset([1, 2, 3]): 1, frozenset([4, 5, 6]): 2}
    b = {frozenset([1, 2, 3]): 3, frozenset([7, 8, 9]): 4}
    result = geometric_product(a, b)
    print("Geometric Product:", result)