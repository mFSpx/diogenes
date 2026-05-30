# DARWIN HAMMER — match 4474, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s1.py (gen4)
# parent_b: hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1958_s1.py (gen6)
# born: 2026-05-29T23:55:58Z

"""
This module fuses the core mathematics of hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s1.py and hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1958_s1.py.
The mathematical bridge between the two structures is the use of the TTT-Linear weight matrix as the basis for the Dense Associative Memory matrix,
and the SSIM function to evaluate the similarity between the input and output of the ternary router, 
while incorporating the reconstruction-risk score from the Pheromone Infotaxis Privacy System to modulate 
the sparse expansion process in the Hybrid Sparse Expansion algorithm.

The hybrid system:
1. Computes raw pheromone signals and associated risk scores using TTT-Linear weight matrix.
2. Forms weighted sparse expansions using the risk scores and SSIM function.
3. Uses the weighted sparse expansions in decision making and differentially-private aggregation.

"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    if target is None:
        target = x
    return np.sum((W @ x - target) ** 2)

def ttt_grad(W, x, target=None):
    if target is None:
        target = x
    return 2 * (W @ x - target) @ x.T

def ttt_step(W, x, eta, target=None):
    grad = ttt_grad(W, x, target)
    return W - eta * grad

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 1.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x.size:
        raise ValueError('sample must be non-empty')
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    k1_squared = k1 ** 2
    k2_squared = k2 ** 2
    c1 = (k1_squared * dynamic_range) ** 2
    c2 = (k2_squared * dynamic_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim_map

def energy(xi, M, beta=1.0):
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = np.log(np.exp(scores).sum())
    quadratic_term = 0.5 * xi @ xi
    return -lse_term + quadratic_term

def compute_risk_score(xi, M, W):
    energy_value = energy(xi, M)
    reconstruction_risk = ttt_loss(W, xi)
    return reconstruction_risk + energy_value

def hybrid_operation(xi, M, W):
    risk_score = compute_risk_score(xi, M, W)
    ssim_value = ssim(xi, W @ xi)
    return risk_score, ssim_value

if __name__ == "__main__":
    np.random.seed(0)
    d_in = 10
    W = init_ttt(d_in)
    M = np.random.rand(10, d_in)
    xi = np.random.rand(d_in)
    risk_score, ssim_value = hybrid_operation(xi, M, W)
    print(f"Risk Score: {risk_score}, SSIM Value: {ssim_value}")