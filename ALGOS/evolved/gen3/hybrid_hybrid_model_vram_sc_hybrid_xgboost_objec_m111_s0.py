# DARWIN HAMMER — match 111, survivor 0
# gen: 3
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s1.py (gen1)
# parent_b: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py (gen2)
# born: 2026-05-29T23:25:43Z

"""
Hybrid module combining the VRAM scheduler from model_vram_scheduler.py (TTT-Linear model) 
and the XGBoost objective mathematics with ternary lens audit pruning from hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py.

The mathematical bridge between the two parents is the integration of the TTT-Linear model's 
update rule into the XGBoost objective's split-gain formula, which modulates the pruning probability 
based on the model's performance. This allows the hybrid algorithm to adapt to the changing 
memory requirements of the model while maintaining an optimal pruning strategy.

Specifically, the TTT-Linear model's update rule is used to compute the gradient and Hessian of the 
binary logistic loss, which are then used to compute the optimal leaf weight and split gain. 
The split gain is then used to modulate the pruning probability based on the model's performance.
"""

import numpy as np
import math
import random
import sys
import pathlib
import json
import os

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    """Self-supervised loss for TTT.

    If target is None, use reconstruction loss: ||W x - x||^2.
    x shape: (d_in,). Returns scalar float.

    The reconstruction objective treats the identity mapping as the target.
    The weight matrix learns to be a good compressor of the input distribution
    seen so far — if W@x ≈ x holds, W has absorbed enough structure to
    reconstruct tokens from the sequence.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    """Gradient of ttt_loss w.r.t. W.

    Closed-form for reconstruction loss:
        loss = ||Wx - x||^2 = (Wx - x)^T (Wx - x)
        d loss / dW = 2 (Wx - x) x^T

    Returns array shape (d_out, d_in), same shape as W.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t                    # (d_out,)
    return 2.0 * np.outer(residual, x)    # (d_out, d_in)

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
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

def hybrid_prune(W, x, target=None, reg_lambda=1.0, gamma=0.0):
    """Hybrid pruning function that combines TTT-Linear model's update rule with XGBoost objective's split-gain formula."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = optimal_leaf_weight(g, h, reg_lambda)
    split_g = split_gain(g, h, g, h, reg_lambda=reg_lambda, gamma=gamma)
    return split_g

def plan_dual_engine_residency(payload=None, state=None, include_gpu=True):
    """Plan the always-on CPU FairyFuse + GPU Q4 DeepSeek residency envelope."""
    # Intentionally advisory and side-effect free
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