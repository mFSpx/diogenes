# DARWIN HAMMER — match 4134, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1969_s2.py (gen4)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s5.py (gen4)
# born: 2026-05-29T23:53:41Z

"""Hybrid RBF‑NLMS‑TTT Algorithm
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1969_s2.py (RBF surrogate + TTT‑Linear)
- hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s5.py (FCD‑modulated NLMS + bandit)

Mathematical Bridge
-------------------
Both parents perform adaptive weight updates:

* Parent A updates a linear weight matrix **W** (TTT‑Linear) while a
  radial‑basis‑function (Gaussian) kernel evaluates similarity between the
  current input **x** and a set of prototype vectors **C**.  The kernel value
  enters the loss as a regulariser that encourages low free‑energy (high
  similarity) configurations.

* Parent B drives an NLMS (Normalized Least‑Mean‑Squares) filter whose step‑size
  μ̂ is scaled by a modulation factor **(1+⟨y⟩)**, where ⟨y⟩ is the average
  fold‑change‑detection (FCD) signal over a recent horizon.  This couples the
  learning‑rate to external dynamics.

The hybrid algorithm fuses these ideas by:

1. Computing a Gaussian RBF similarity **k(x, C)** between the input **x**
   and prototype set **C** (Parent A).
2. Forming a composite loss  

   L(W) = ‖W x − t‖² + λ · (1 − k(x, C))²  

   where **t** is the target, λ controls regularisation and the kernel term
   penalises low similarity.
3. Updating **W** with an NLMS‑style rule whose effective step‑size is  

   μ_eff = μ₀ · (1 + ⟨y⟩) · exp(−α · ε)  

   – μ₀ : base learning rate,  
   – ⟨y⟩ : average FCD signal (Parent B),  
   – ε : kernel bandwidth (Parent A),  
   – α : scaling constant.

Thus the RBF kernel supplies a data‑driven regulariser while the FCD signal
modulates how aggressively the weights adapt, yielding a unified adaptive
system that respects both free‑energy minimisation and temporal context.

The module provides three core functions that demonstrate this hybrid
operation and a smoke‑test runnable as ``python hybrid_rbf_nlms_ttt.py``.
"""

import math
import random
import sys
from pathlib import Path
from typing import Sequence, List

import numpy as np

Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel ϕ(r)=exp(-(ε r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_similarity(x: np.ndarray, centers: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Compute the average Gaussian RBF similarity between a vector ``x`` and a
    collection of prototype vectors ``centers``.
    """
    if centers.ndim != 2:
        raise ValueError("centers must be a 2‑D array (n_centers, dim)")
    dists = np.linalg.norm(centers - x, axis=1)
    sims = np.vectorize(lambda r: gaussian(r, epsilon))(dists)
    return float(np.mean(sims))


def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix **W** with small random entries."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def fold_change(series: np.ndarray) -> np.ndarray:
    """
    Simple Fold‑Change Detection (FCD).
    For a scalar time series ``s`` it returns y_t = log(s_t / s_{t‑1}),
    with a small epsilon to avoid division by zero.
    """
    eps = 1e-8
    return np.log((series[1:] + eps) / (series[:-1] + eps))


def fcd_average(y: np.ndarray) -> float:
    """Return the mean of the FCD signal (used as modulation ⟨y⟩)."""
    if y.size == 0:
        return 0.0
    return float(np.mean(y))


def hybrid_loss(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    centers: np.ndarray,
    epsilon: float,
    lam: float = 0.1,
) -> float:
    """
    Composite loss L(W) = ‖W x − t‖² + λ·(1 − k)²,
    where k is the average RBF similarity between ``x`` and ``centers``.
    """
    pred = W @ x
    mse = np.mean((pred - target) ** 2)
    k = rbf_similarity(x, centers, epsilon)
    reg = lam * (1.0 - k) ** 2
    return mse + reg


def nlms_update(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    mu_base: float,
    y_mod: float,
    epsilon: float,
    lam: float = 0.1,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Perform a single NLMS‑style update on ``W`` using the composite loss.
    The effective step size is

        μ_eff = μ_base * (1 + y_mod) * exp(-α·ε)

    ``y_mod`` is the average FCD signal ⟨y⟩, and ``ε`` is the RBF bandwidth.
    """
    # prediction and error
    pred = W @ x
    err = target - pred  # shape (d_out,)

    # NLMS normalisation term (avoid division by zero)
    norm = np.dot(x, x) + 1e-12

    # effective learning rate
    mu_eff = mu_base * (1.0 + y_mod) * math.exp(-alpha * epsilon)

    # Gradient of the MSE part: -2 * err * xᵀ  (but NLMS uses err·xᵀ / ||x||²)
    delta_mse = (mu_eff / norm) * np.outer(err, x)

    # Gradient of the RBF regulariser:
    # ∂/∂W (λ·(1−k)²) = -2·λ·(1−k)·∂k/∂W
    # where k = avg_i exp(-(ε·‖x−c_i‖)²).  ∂k/∂W = 0 because k does not depend on W,
    # but we still include a small corrective term that nudges W towards
    # reducing the distance between x and the nearest centre (proxy regularisation).
    # Here we use a simple proxy: move W in the direction of (c_nearest - x).
    dists = np.linalg.norm(centers - x, axis=1)
    nearest_idx = int(np.argmin(dists))
    c_nearest = centers[nearest_idx]
    proxy = (c_nearest - x)  # shape (dim,)
    delta_rbf = -2.0 * lam * (1.0 - rbf_similarity(x, centers, epsilon)) * np.outer(
        np.ones(W.shape[0]), proxy
    ) * (mu_eff / norm)

    # Total update
    W_new = W + delta_mse + delta_rbf
    return W_new


def hybrid_step(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    centers: np.ndarray,
    epsilon: float,
    mu_base: float,
    fcd_series: np.ndarray,
    lam: float = 0.1,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Execute one hybrid adaptation step:
      1. Compute the FCD modulation ⟨y⟩ from ``fcd_series``.
      2. Update ``W`` with NLMS‑style rule that also respects the RBF regulariser.
    Returns the updated weight matrix.
    """
    # 1. Fold‑Change detection on the provided series
    y_signal = fold_change(fcd_series)
    y_mod = fcd_average(y_signal)

    # 2. Perform the combined update
    W_upd = nlms_update(
        W,
        x,
        target,
        mu_base=mu_base,
        y_mod=y_mod,
        epsilon=epsilon,
        lam=lam,
        alpha=alpha,
    )
    return W_upd


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic problem: map 5‑dim input to 3‑dim output.
    dim_in = 5
    dim_out = 3
    np.random.seed(42)

    # Initialise weight matrix (TTT‑Linear style)
    W = init_ttt(dim_in, dim_out, scale=0.05, seed=1)

    # Random prototype centres for the RBF kernel
    n_centers = 8
    centers = np.random.randn(n_centers, dim_in)

    # Generate a random input vector and a target output
    x = np.random.randn(dim_in)
    target = np.random.randn(dim_out)

    # Simulated external signal for FCD (e.g., a slowly varying scalar)
    external_signal = np.cumsum(np.random.randn(50) * 0.01) + 1.0

    # Hyper‑parameters
    epsilon = 0.8      # RBF bandwidth
    mu_base = 0.2      # Base NLMS step size
    lam = 0.05         # Regularisation weight
    alpha = 0.3        # Kernel‑learning‑rate coupling

    # Perform a single hybrid adaptation step
    W_new = hybrid_step(
        W,
        x,
        target,
        centers,
        epsilon,
        mu_base,
        external_signal,
        lam=lam,
        alpha=alpha,
    )

    # Print diagnostics to verify execution
    print("Initial W norm :", np.linalg.norm(W))
    print("Updated W norm :", np.linalg.norm(W_new))
    print("Loss before    :", hybrid_loss(W, x, target, centers, epsilon, lam))
    print("Loss after     :", hybrid_loss(W_new, x, target, centers, epsilon, lam))
    print("Test completed without error.")