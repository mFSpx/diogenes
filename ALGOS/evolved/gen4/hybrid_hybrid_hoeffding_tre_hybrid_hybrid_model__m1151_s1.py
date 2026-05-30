# DARWIN HAMMER — match 1151, survivor 1
# gen: 4
# parent_a: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s1.py (gen3)
# born: 2026-05-29T23:33:03Z

"""
This module integrates the Hoeffding bound helpers for stream splits from hoeffding_tree_tropical_maxplus_m18_s1 and 
the hybrid xgboost objective from hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s1. 
The mathematical bridge between these structures is found in the application of tropical polynomials to model decision boundaries 
in ReLU networks and the use of Hoeffding bounds to guide the splitting process, which can be optimized using the hybrid xgboost objective.
By leveraging the Hoeffding bound to minimize the impact of noise in the data stream and the hybrid xgboost objective to optimize the splitting process, 
we can create a more robust and efficient decision-making system.
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

def relu_layer_to_tropical(W, b):
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    return W.copy(), b.copy()

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(
    gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0
) -> float:
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))

def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0
) -> float:
    return 0.5 * (
        (left_gradient ** 2) / (left_hessian + reg_lambda)
        + (right_gradient ** 2) / (right_hessian + reg_lambda)
        - (left_gradient + right_gradient) ** 2 / (left_hessian + right_hessian + reg_lambda)
    ) - gamma

def hybrid_prune(W, x, target=None, reg_lambda=1.0, gamma=0.0, learning_rate=0.1):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = optimal_leaf_weight(g, h, reg_lambda)
    W_update = W - learning_rate * 2.0 * np.outer(residual, x)
    split_g = split_gain(g, h, g, h, reg_lambda=reg_lambda, gamma=gamma)
    return W_update, split_g

def hybrid_tropical_prune(W, x, target=None, reg_lambda=1.0, gamma=0.0, learning_rate=0.1):
    pred = t_matmul(W, x)
    t = x if target is None else target
    residual = t_add(pred, -t)
    margin = residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = optimal_leaf_weight(g, h, reg_lambda)
    W_update = W - learning_rate * 2.0 * np.outer(residual, x)
    split_g = split_gain(g, h, g, h, reg_lambda=reg_lambda, gamma=gamma)
    return W_update, split_g

def hybrid_hoeffding_prune(W, x, target=None, reg_lambda=1.0, gamma=0.0, learning_rate=0.1, r=1.0, delta=0.1, n=1):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = optimal_leaf_weight(g, h, reg_lambda)
    eps = hoeffding_bound(r, delta, n)
    W_update = W - learning_rate * 2.0 * np.outer(residual, x)
    split_g = split_gain(g, h, g, h, reg_lambda=reg_lambda, gamma=gamma)
    return W_update, split_g, eps

if __name__ == "__main__":
    d_in = 10
    d_out = 10
    scale = 0.01
    seed = 0
    W = np.random.default_rng(seed).standard_normal((d_out, d_in)) * scale
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    reg_lambda = 1.0
    gamma = 0.0
    learning_rate = 0.1
    W_update, split_g = hybrid_prune(W, x, target, reg_lambda, gamma, learning_rate)
    W_update_t, split_g_t = hybrid_tropical_prune(np.random.rand(1, 1, d_in), np.random.rand(d_in), target)
    W_update_h, split_g_h, eps = hybrid_hoeffding_prune(W, x, target, reg_lambda, gamma, learning_rate, 1.0, 0.1, 1)