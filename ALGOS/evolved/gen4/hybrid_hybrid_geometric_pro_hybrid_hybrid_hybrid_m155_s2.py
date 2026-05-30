# DARWIN HAMMER — match 155, survivor 2
# gen: 4
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py (gen3)
# born: 2026-05-29T23:27:21Z

import numpy as np
import math
import random
import sys
import pathlib
from typing import Any, Dict, Tuple

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
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
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    """Self-supervised loss for TTT."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    """Gradient of ttt_loss w.r.t. W."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM) between two 1D arrays.
    """
    # Compute means
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    # Compute variances
    sigma_x = np.std(x)
    sigma_y = np.std(y)

    # Compute covariance
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    # Compute SSIM
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))

    return ssim_val

def geometric_product(a, b):
    """Full Clifford product ab."""
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            if blade_out in result:
                result[blade_out] += coef_a * coef_b * sign
            else:
                result[blade_out] = coef_a * coef_b * sign
    return result

def hybrid_step(W, x, alpha=0.1, beta=0.1, gamma=0.001):
    """Update W using the combined loss."""
    pred = W @ x
    residual = pred - x
    ssim_val = ssim(pred, x)
    loss_rec = float(residual @ residual)
    loss_ssim = 1 - ssim_val
    loss_hybrid = alpha * loss_rec + beta * loss_ssim + gamma * np.linalg.norm(W)
    grad_W = 2.0 * np.outer(residual, x) + beta * np.outer((pred - x), x) * (loss_ssim / np.linalg.norm(pred - x)) + 2 * gamma * W
    W -= 0.01 * grad_W
    return W, loss_hybrid

def hybrid_forward(W, x):
    """Apply the current W to an input vector."""
    return W @ x

def multi_step(W, x, n_steps=10, alpha=0.1, beta=0.1, gamma=0.001):
    """Perform multiple hybrid steps."""
    losses = []
    for _ in range(n_steps):
        W, loss = hybrid_step(W, x, alpha, beta, gamma)
        losses.append(loss)
    return W, losses

if __name__ == "__main__":
    np.random.seed(0)
    W = init_ttt(10)
    x = np.random.rand(10)
    W, losses = multi_step(W, x)
    print(f"Losses: {losses}")
    pred = hybrid_forward(W, x)
    print(f"Prediction: {pred}")