# DARWIN HAMMER — match 2746, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2218_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s1.py (gen5)
# born: 2026-05-29T23:45:37Z

"""Hybrid RLCT‑Guided Geometric‑Product Training with Pheromone‑Weighted Updates
==========================================================================

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_rlct_g_hybrid_cockpit_metri_m457_s1.py`
  provides a Real Log‑Canonical Threshold (RLCT) estimator derived from training
  losses and a mechanism to weight pheromone signals by an *honesty* factor.
* **Parent B** – `hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s1.py`
  supplies a Clifford‑algebra geometric product, a test‑time‑training (TTT) loss,
  a stylometry‑hash regularizer and their combined gradient update.

Mathematical Bridge
-------------------
The RLCT λ obtained from Parent A quantifies the *complexity* of the learning
trajectory.  We use λ to re‑scale the coefficients of the unified objective  

\[
L_{\text{hyb}} = \alpha(\lambda)\,L_{\text{TTT}}
               + \beta(\lambda)\,L_{\text{hash}}
               + \gamma\,L_{\text{SSIM}},
\]

where  

* \(\alpha(\lambda)=\frac{1}{1+\lambda}\)  (larger λ → weaker TTT pressure),  
* \(\beta(\lambda)=\frac{\lambda}{1+\lambda}\) (larger λ → stronger hash regularisation),  
* \(\gamma\) is a fixed small constant.

The gradient of \(L_{\text{hyb}}\) is then modulated element‑wise by the
*honesty‑weighted pheromone signal* supplied by Parent A, providing a biologically
inspired bias on each weight update.

The code below implements:
1. RLCT estimation (`estimate_rlct_from_losses`).
2. Clifford‑algebra geometric product (`geometric_product`).
3. Hybrid loss and its gradient (`hybrid_loss`, `hybrid_gradient`).
4. Pheromone‑aware weight update (`hybrid_update`).

All required imports are from the Python standard library and NumPy only.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict

# ----------------------------------------------------------------------
# Parent A – RLCT and pheromone utilities
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses_per_n, n_values):
    """Estimate the Real Log‑Canonical Threshold (RLCT) from loss vs. sample size.

    Implements the linear regression of log(loss) on log(log(n)) as described
    in the original parent algorithm.
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= np.e):
        raise ValueError("All n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))

    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    slope = float((x_c * y_c).sum() / var_x)
    return slope  # This slope is the RLCT λ


def anti_slop_ratio(claims_with_evidence, total_claims_emitted):
    """Simple honesty metric: proportion of claims backed by evidence."""
    if total_claims_emitted <= 0:
        return 0.0
    return min(1.0, max(0.0, claims_with_evidence / total_claims_emitted))


def decay_factor(elapsed_seconds, half_life_seconds):
    """Exponential decay based on half‑life."""
    if half_life_seconds <= 0:
        return 0.0
    return 0.5 ** (elapsed_seconds / half_life_seconds)


class PheromoneSystem:
    """Stores pheromone signals and provides honesty‑weighted, time‑decayed values."""

    def __init__(self):
        # Keys are generic identifiers (e.g., weight indices); values are dicts
        self.signals: Dict[Tuple[int, int], Dict] = {}

    def emit(self, key, signal_value, half_life_seconds,
             claims_with_evidence, total_claims_emitted):
        """Record a new pheromone emission."""
        now = datetime.now(timezone.utc)
        honesty = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
        weighted = signal_value * honesty
        self.signals[key] = {
            "value": weighted,
            "timestamp": now,
            "half_life": half_life_seconds,
        }

    def get(self, key):
        """Retrieve the current decayed pheromone value for *key*."""
        entry = self.signals.get(key)
        if entry is None:
            return 1.0  # neutral multiplicative factor
        now = datetime.now(timezone.utc)
        elapsed = (now - entry["timestamp"]).total_seconds()
        factor = decay_factor(elapsed, entry["half_life"])
        return entry["value"] * factor if factor > 0 else 0.0


# ----------------------------------------------------------------------
# Parent B – Geometric product and hybrid loss utilities
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Simplified geometric product for vectors.

    For vectors a, b ∈ ℝⁿ the geometric product G = a·b + a∧b is represented as
    a matrix whose symmetric part is the inner product and whose antisymmetric
    part encodes the outer (wedge) product.
    """
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    inner = np.dot(a.T, b)  # scalar (1×1) matrix
    outer = np.dot(a, b)    # n×n matrix (outer product)
    # Symmetric part (inner) added to diagonal of outer product
    G = outer + np.eye(a.shape[0]) * inner
    return G


def hybrid_loss(W: np.ndarray, x: np.ndarray, H: np.ndarray,
                rlct: float, gamma: float = 0.5) -> float:
    """Unified loss mixing TTT, hash regularisation and SSIM‑like term.

    Parameters
    ----------
    W : (d, d) weight matrix.
    x : (d,) input vector (test‑time sample).
    H : (d, d) stylometry‑hash target matrix.
    rlct : scalar λ from RLCT estimation.
    gamma : fixed coefficient for the SSIM‑like component.
    """
    # Coefficients derived from RLCT
    alpha = 1.0 / (1.0 + rlct)
    beta = rlct / (1.0 + rlct)

    # TTT loss: ||W x - x||²
    diff = W @ x - x
    L_TTT = np.dot(diff, diff)

    # Hash regularisation: Frobenius norm ||W - H||_F²
    L_hash = np.linalg.norm(W - H, ord="fro") ** 2

    # SSIM‑like term on geometric products
    gp_pred = geometric_product(W @ x, x)
    gp_true = geometric_product(x, x)
    L_SSIM = np.linalg.norm(gp_pred - gp_true, ord="fro") ** 2

    return alpha * L_TTT + beta * L_hash + gamma * L_SSIM


def hybrid_gradient(W: np.ndarray, x: np.ndarray, H: np.ndarray,
                    rlct: float, gamma: float = 0.5) -> np.ndarray:
    """Gradient of `hybrid_loss` w.r.t. the weight matrix W."""
    alpha = 1.0 / (1.0 + rlct)
    beta = rlct / (1.0 + rlct)

    # ----- Gradient of TTT term -----
    # L_TTT = ||W x - x||²  →  ∂L/∂W = 2 (W x - x) xᵀ
    diff = W @ x - x
    grad_TTT = 2.0 * np.outer(diff, x)

    # ----- Gradient of hash regulariser -----
    # L_hash = ||W - H||_F²  →  ∂L/∂W = 2 (W - H)
    grad_hash = 2.0 * (W - H)

    # ----- Gradient of SSIM‑like term -----
    # Approximate by treating geometric_product as linear in its first argument:
    # G = geometric_product(Wx, x) ≈ (Wx) xᵀ + x (Wx)ᵀ  (symmetrised outer product)
    Wx = W @ x
    gp_pred = geometric_product(Wx, x)
    gp_true = geometric_product(x, x)
    diff_gp = gp_pred - gp_true
    # Derivative w.r.t. Wx is (diff_gp + diff_gpᵀ) * xᵀ
    sym = diff_gp + diff_gp.T
    grad_Wx = sym @ x[:, None]  # shape (d,1)
    # Chain rule: ∂Wx/∂W = xᵀ
    grad_SSIM = grad_Wx @ x[None, :]  # shape (d,d)
    grad_SSIM = 2.0 * grad_SSIM  # match squared norm derivative

    # Combine with coefficients
    return alpha * grad_TTT + beta * grad_hash + gamma * grad_SSIM


def hybrid_update(W: np.ndarray, x: np.ndarray, H: np.ndarray,
                  rlct: float, pheromone: PheromoneSystem,
                  learning_rate: float = 0.01,
                  gamma: float = 0.5) -> np.ndarray:
    """Perform a single honesty‑weighted, pheromone‑modulated gradient step."""
    grad = hybrid_gradient(W, x, H, rlct, gamma)

    # Apply pheromone scaling element‑wise
    d, _ = W.shape
    scaled_grad = np.empty_like(W)
    for i in range(d):
        for j in range(d):
            factor = pheromone.get((i, j))
            # Ensure a neutral factor when no pheromone exists
            factor = factor if factor > 0 else 1.0
            scaled_grad[i, j] = grad[i, j] * factor

    # Gradient descent update
    W_new = W - learning_rate * scaled_grad
    return W_new


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    d = 4
    rng = np.random.default_rng(42)

    W = rng.normal(scale=0.1, size=(d, d))
    x = rng.normal(size=d)
    H = rng.normal(scale=0.1, size=(d, d))

    # Simulated loss curve for RLCT estimation
    n_vals = np.arange(30, 200, 10)
    losses = 1.0 / np.sqrt(n_vals) + rng.normal(scale=0.02, size=n_vals.shape)

    rlct = estimate_rlct_from_losses(losses, n_vals)
    print(f"Estimated RLCT λ = {rlct:.4f}")

    # Initialise pheromone system and emit a few signals
    pher = PheromoneSystem()
    half_life = 60.0  # seconds
    for i in range(d):
        for j in range(d):
            signal = random.random()
            claims = random.randint(0, 5)
            total = random.randint(5, 10)
            pher.emit((i, j), signal, half_life, claims, total)

    # Compute initial loss
    loss_before = hybrid_loss(W, x, H, rlct)
    print(f"Loss before update: {loss_before:.6f}")

    # Perform an update
    W_updated = hybrid_update(W, x, H, rlct, pheromone=pher, learning_rate=0.05)

    # Compute loss after update
    loss_after = hybrid_loss(W_updated, x, H, rlct)
    print(f"Loss after update:  {loss_after:.6f}")

    # Verify that the code runs without exception
    assert not np.isnan(loss_after), "Loss became NaN after update"
    print("Smoke test completed successfully.")