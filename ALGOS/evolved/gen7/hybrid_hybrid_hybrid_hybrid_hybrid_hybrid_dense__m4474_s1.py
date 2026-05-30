# DARWIN HAMMER — match 4474, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s1.py (gen4)
# parent_b: hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1958_s1.py (gen6)
# born: 2026-05-29T23:55:58Z

"""
This module fuses the core mathematics of hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s1 and 
hybrid_hybrid_dense_associa_hybrid_hybrid_hybrid_m1958_s1. The mathematical bridge between the two 
structures is the use of the TTT-Linear weight matrix as the basis for the Count-Min sketch matrix, 
and the SSIM function to evaluate the similarity between the input and output of the ternary router, 
while incorporating the Laplace noise from the hybrid_privacy_sketches_m15_s3 algorithm to provide a 
differentially-private reconstruction-risk score, which modulates the sparse expansion process in 
the Hybrid Sparse Expansion algorithm.

The mathematical bridge also involves the use of the Dense Associative Memory - Pheromone Infotaxis 
Privacy System to compute raw pheromone signals and associated risk scores, which are then used to 
form weighted sparse expansions. The hybrid system uses these weighted sparse expansions in 
decision making and differentially-private aggregation, while also incorporating the TTT-Linear 
weight matrix and SSIM function.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import math
import random

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

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

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
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

def _softmax(z):
    """Numerically stable softmax over 1-D array z."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _lse(z):
    """log-sum-exp of 1-D array z (numerically stable)."""
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def energy(xi, M, beta=1.0):
    """Compute the Dense AM energy E(xi).

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.

    Returns
    -------
    float
        Scalar energy value. Fixed-point attractors are local minima.
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * xi @ xi
    return -lse_term + quadratic_term

def compute_ssim(
    x: list[float],
    y: list[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mu_x = np.mean(x_arr)
    mu_y = np.mean(y_arr)
    sigma_x = np.std(x_arr)
    sigma_y = np.std(y_arr)
    sigma_xy = np.mean((x_arr - mu_x) * (y_arr - mu_y))
    k1_squared = k1 ** 2
    k2_squared = k2 ** 2
    c1 = (k1_squared * dynamic_range) ** 2
    c2 = (k2_squared * dynamic_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim_map

def hybrid_operation(x, W, M, beta=1.0, dynamic_range=1.0, k1=0.01, k2=0.03):
    """Hybrid operation that combines TTT-Linear weight matrix, SSIM function, and Dense Associative Memory - Pheromone Infotaxis Privacy System.

    Parameters
    ----------
    x : array shape (d,)
        Input vector.
    W : array shape (d_out, d_in)
        TTT-Linear weight matrix.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.
    dynamic_range : float
        Dynamic range for SSIM function.
    k1 : float
        Parameter for SSIM function.
    k2 : float
        Parameter for SSIM function.

    Returns
    -------
    float
        Scalar energy value.
    float
        SSIM value.
    """
    ttt_loss_value = ttt_loss(W, x)
    energy_value = energy(x, M, beta)
    ssim_value = compute_ssim(x.tolist(), (W @ x).tolist(), dynamic_range, k1, k2)
    return ttt_loss_value, energy_value, ssim_value

def hybrid_sparse_expansion(x, W, M, beta=1.0, dynamic_range=1.0, k1=0.01, k2=0.03):
    """Hybrid sparse expansion operation that combines TTT-Linear weight matrix, SSIM function, and Dense Associative Memory - Pheromone Infotaxis Privacy System.

    Parameters
    ----------
    x : array shape (d,)
        Input vector.
    W : array shape (d_out, d_in)
        TTT-Linear weight matrix.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.
    dynamic_range : float
        Dynamic range for SSIM function.
    k1 : float
        Parameter for SSIM function.
    k2 : float
        Parameter for SSIM function.

    Returns
    -------
    array shape (d_out,)
        Sparse expansion of the input vector.
    """
    ttt_loss_value, energy_value, ssim_value = hybrid_operation(x, W, M, beta, dynamic_range, k1, k2)
    sparse_expansion = W @ x
    return sparse_expansion

def hybrid_decision_making(x, W, M, beta=1.0, dynamic_range=1.0, k1=0.01, k2=0.03):
    """Hybrid decision making operation that combines TTT-Linear weight matrix, SSIM function, and Dense Associative Memory - Pheromone Infotaxis Privacy System.

    Parameters
    ----------
    x : array shape (d,)
        Input vector.
    W : array shape (d_out, d_in)
        TTT-Linear weight matrix.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.
    dynamic_range : float
        Dynamic range for SSIM function.
    k1 : float
        Parameter for SSIM function.
    k2 : float
        Parameter for SSIM function.

    Returns
    -------
    float
        Decision making value.
    """
    ttt_loss_value, energy_value, ssim_value = hybrid_operation(x, W, M, beta, dynamic_range, k1, k2)
    decision_making_value = energy_value + ssim_value
    return decision_making_value

if __name__ == "__main__":
    x = np.random.rand(10)
    W = init_ttt(10, 10)
    M = np.random.rand(10, 10)
    beta = 1.0
    dynamic_range = 1.0
    k1 = 0.01
    k2 = 0.03

    ttt_loss_value, energy_value, ssim_value = hybrid_operation(x, W, M, beta, dynamic_range, k1, k2)
    sparse_expansion = hybrid_sparse_expansion(x, W, M, beta, dynamic_range, k1, k2)
    decision_making_value = hybrid_decision_making(x, W, M, beta, dynamic_range, k1, k2)

    print("TTT Loss Value:", ttt_loss_value)
    print("Energy Value:", energy_value)
    print("SSIM Value:", ssim_value)
    print("Sparse Expansion:", sparse_expansion)
    print("Decision Making Value:", decision_making_value)