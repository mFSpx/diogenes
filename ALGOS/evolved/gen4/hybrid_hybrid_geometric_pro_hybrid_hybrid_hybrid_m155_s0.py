# DARWIN HAMMER — match 155, survivor 0
# gen: 4
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py (gen3)
# born: 2026-05-29T23:27:21Z

"""
Hybrid Algorithm: Geometric Product Guided Test-Time Training (Hybrid-GP-TTT)

Parents
-------
* **Parent A** – `hybrid_geometric_product_hybrid_model_vram_sc_m22_s0.py`
  provides a geometric product implementation, which can be viewed as a form of optimization problem.
* **Parent B** – `hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py`
  implements a Test-Time Training (TTT) algorithm with a Structural Similarity (SSIM) function.

Mathematical Bridge
-------------------
The mathematical bridge between the two parents is the update rule of the TTT algorithm, which can be integrated with the geometric product's blade arithmetic.
By using the SSIM loss as a regularization term in the geometric product optimization problem, we can create a hybrid algorithm that adapts to the changing requirements of the model.
The hybrid algorithm treats `L_hybrid = α·L_rec + β·L_ssim` as a unified objective, where `L_rec` is the reconstruction error and `L_ssim` is the SSIM loss.
"""

import numpy as np
import math
import random
import sys
import pathlib

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
            if blade_out in result:
                result[blade_out] += coef_a * coef_b * sign
            else:
                result[blade_out] = coef_a * coef_b * sign
    return result

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))

def hybrid_loss(W, x, target=None, alpha=0.5, beta=0.5):
    rec_loss = ttt_loss(W, x, target)
    ssim_loss = 1 - ssim(x, W @ x)
    return alpha * rec_loss + beta * ssim_loss

def hybrid_step(W, x, target=None, alpha=0.5, beta=0.5, learning_rate=0.01):
    loss = hybrid_loss(W, x, target, alpha, beta)
    grad = ttt_grad(W, x, target) + beta * ttt_grad(W, x, target) * (1 - ssim(x, W @ x))
    return W - learning_rate * grad

def hybrid_forward(W, x):
    return W @ x

if __name__ == "__main__":
    W = init_ttt(10)
    x = np.random.rand(10)
    target = np.random.rand(10)
    print(hybrid_loss(W, x, target))
    W = hybrid_step(W, x, target)
    print(hybrid_forward(W, x))