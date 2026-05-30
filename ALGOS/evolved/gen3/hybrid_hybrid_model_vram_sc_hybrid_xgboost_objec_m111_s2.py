# DARWIN HAMMER — match 111, survivor 2
# gen: 3
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s1.py (gen1)
# parent_b: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py (gen2)
# born: 2026-05-29T23:25:43Z

import numpy as np
import math
import random
import sys
import pathlib
import json
import os

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

def hybrid_prune(W, x, target=None, reg_lambda=1.0, gamma=0.0, delta=0.001):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = optimal_leaf_weight(g, h, reg_lambda)
    split_g = split_gain(g, h, g, h, reg_lambda=reg_lambda, gamma=gamma)
    if abs(split_g) < delta:
        return 0.0
    else:
        return split_g

def plan_dual_engine_residency(payload=None, state=None, include_gpu=True):
    pass

if __name__ == "__main__":
    d_in = 10
    d_out = 10
    scale = 0.01
    seed = 0
    W = init_ttt(d_in, d_out, scale, seed)
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    reg_lambda = 1.0
    gamma = 0.0
    split_g = hybrid_prune(W, x, target, reg_lambda, gamma)
    print("Split gain:", split_g)

    # Test the improved hybrid_prune function with different inputs
    for _ in range(10):
        x = np.random.rand(d_in)
        target = np.random.rand(d_in)
        split_g = hybrid_prune(W, x, target, reg_lambda, gamma)
        print("Split gain:", split_g)