# DARWIN HAMMER — match 1151, survivor 0
# gen: 4
# parent_a: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s1.py (gen3)
# born: 2026-05-29T23:33:03Z

"""
This module fuses the Hoeffding bound-based decision tree splitting from hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py
and the TTT (ternary tensor train) based hybrid model from hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s1.py.
The mathematical bridge between these structures is found in the application of tropical polynomials to model decision boundaries,
which informs the decision to split in Hoeffding trees. The TTT model's ternary weights are used to construct tropical polynomials,
which are then evaluated using tropical polynomial operations to guide the splitting process.

Author: [Your Name]
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

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

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def hybrid_split_decision(W, x, target, r, delta, n):
    ttt_pred = W @ x
    loss = ttt_loss(W, x, target)
    grad = ttt_grad(W, x, target)
    tropical_poly = t_polyval(np.diag(W), x)
    gain = -loss + tropical_poly
    second_best_gain = -loss
    return should_split(gain, second_best_gain, r, delta, n)

def tropical_split(W, x):
    ttt_pred = W @ x
    tropical_poly = t_polyval(np.diag(W), x)
    return ttt_pred, tropical_poly

def hybrid_prune(W, x, target=None, reg_lambda=1.0, gamma=0.0, learning_rate=0.1):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g = 2 * residual
    h = 2 * np.ones_like(residual)
    optimal_weight = -g / (h + reg_lambda)
    W_update = W - learning_rate * np.outer(g, x)
    split_g = 0.5 * ((g ** 2) / (h + reg_lambda)) - gamma
    return W_update, split_g

if __name__ == "__main__":
    d_in = 10
    d_out = 10
    scale = 0.01
    seed = 0
    W = init_ttt(d_in, d_out, scale, seed)
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    r = 1.0
    delta = 0.1
    n = 100
    decision = hybrid_split_decision(W, x, target, r, delta, n)
    print(decision)
    ttt_pred, tropical_poly = tropical_split(W, x)
    print(ttt_pred, tropical_poly)
    W_updated, split_g = hybrid_prune(W, x, target)
    print(W_updated, split_g)