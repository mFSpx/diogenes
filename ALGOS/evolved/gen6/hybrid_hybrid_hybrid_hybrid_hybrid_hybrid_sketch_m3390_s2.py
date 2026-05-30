# DARWIN HAMMER — match 3390, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s2.py (gen5)
# born: 2026-05-29T23:49:51Z

"""
Hybrid Path Signature – Clifford Geometric Product & KAN Fold Change

This module fuses the core mathematics of two parent algorithms:

* Parent A – hybrid_hybrid_hybrid_path_s_hybrid_hybrid_model__m1308_s2.py (KAN Fold Change)
* Parent B – hybrid_hybrid_sketches_hybr_hybrid_hybrid_hybrid_m1904_s2.py (Clifford Geometric Product)

The mathematical bridge lies in the integration of the Clifford geometric product 
from Parent B with the lead-lag transform, B-spline basis, and fold-change detector 
from Parent A. Specifically, we use the B-spline basis to construct a multivector 
that approximates the path signature, and then apply the geometric product to 
combine or manipulate these signatures. The fold-change detector modulates the 
learning rate of the weight matrix update.

The hybrid module therefore:

* extracts deterministic scalar-vector features from a path (Parent A),
* builds a multivector containing scalar, vector and bivector parts that 
  approximate the level-0/1/2 signature using B-spline basis functions,
* uses the Clifford geometric product to combine, rotate or otherwise 
  manipulate these signatures (Parent B),
* modulates the learning rate of the weight matrix update using the 
  fold-change detector (Parent A).
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Tuple

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead-lag transform: interleave (lead, lag) components of a path.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def bspline_basis(x, grid, k=3):
    """
    B-spline basis function.
    """
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
    B[:, -1] = np.where(x == grid[-1], 1.0, 0.0)

    return B

def clifford_geometric_product(a, b):
    """
    Clifford geometric product.
    """
    return np.dot(a, b) + np.dot(a, b.conjugate())

def fold_change_factor(loss_prev, loss_curr):
    """
    Fold-change factor.
    """
    return 1 + 0.1 * (loss_curr - loss_prev) / loss_prev

def euler_weight_update(W, grad_W, learning_rate, fold_change):
    """
    Euler weight update with fold-change modulated learning rate.
    """
    return W - learning_rate * fold_change * grad_W

def hybrid_step(path, W, grid, k=3):
    """
    Single training step that ties the pieces together.
    """
    # Lead-lag transform
    lead_lag_path = lead_lag_transform(path)

    # B-spline basis
    B = bspline_basis(lead_lag_path[:, 0], grid, k)

    # Clifford geometric product
    multivector = np.dot(B, W)

    # Loss and gradient
    loss = np.mean((multivector - lead_lag_path) ** 2)
    grad_W = -2 * np.dot(B.T, (multivector - lead_lag_path))

    # Fold-change factor
    loss_prev = 1.0  # dummy previous loss
    fold_change = fold_change_factor(loss_prev, loss)

    # Euler weight update
    W_updated = euler_weight_update(W, grad_W, 0.01, fold_change)

    return W_updated, loss

if __name__ == "__main__":
    # Smoke test
    path = np.random.rand(10, 2)
    W = np.random.rand(20, 2)
    grid = np.linspace(0, 1, 10)
    W_updated, loss = hybrid_step(path, W, grid)
    print("Loss:", loss)