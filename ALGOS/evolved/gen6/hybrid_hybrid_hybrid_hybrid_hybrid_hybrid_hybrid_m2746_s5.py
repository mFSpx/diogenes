# DARWIN HAMMER — match 2746, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2218_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m675_s1.py (gen5)
# born: 2026-05-29T23:45:37Z

"""Hybrid Fusion Module
=====================

This module fuses the mathematical cores of two parent algorithms:

* **Parent A** – provides a Real Log Canonical Threshold (RLCT) estimator based on
  training losses and an *honesty‑weighted* pheromone signal.
* **Parent B** – defines a geometric‑product‑driven Test‑Time Training (TTT) loss,
  a stylometry‑hash regularizer, and an SSIM component, all acting on a weight
  matrix **W**.

**Mathematical Bridge**

The RLCT value `λ` obtained from Parent A is used to *scale* the three loss
coefficients (α, β, γ) of Parent B:

\[
\alpha = \frac{1}{1 + e^{-\lambda}},\qquad
\beta  = \frac{1}{1 + e^{\lambda}},\qquad
\gamma = \frac{|\lambda|}{1 + |\lambda|}.
\]

The *honesty weight* derived from pheromone evidence further modulates β,
making the hash regularizer stronger when the system is deemed more
trustworthy.

Thus the unified objective is

\[
L_{\text{hyb}} = \alpha L_{\text{TTT}} +
                 \beta\,\omega_{\text{hon}}\,L_{\text{hash}} +
                 \gamma L_{\text{SSIM}} .
\]

The gradient of `L_hyb` is the weighted sum of the individual gradients,
allowing a single update step that simultaneously respects the RLCT‑driven
trust dynamics and the Clifford‑algebra geometric product of Parent B.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime
from typing import Dict, Tuple

# ----------------------------------------------------------------------
# Parent A – RLCT estimation and honesty‑weighted pheromone signal
# ----------------------------------------------------------------------
def estimate_rlct(train_losses_per_n: list, n_values: list) -> float:
    """Estimate the Real Log Canonical Threshold (RLCT) from loss curves.

    Parameters
    ----------
    train_losses_per_n : list of float
        Training loss observed for each sample size ``n``.
    n_values : list of int
        Corresponding sample sizes (must be > e).

    Returns
    -------
    float
        Estimated RLCT λ.
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = (x_centered ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")

    return float((x_centered * y_centered).sum() / var_x)


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Return a ratio in [0,1] reflecting the honesty of the claim source."""
    if total_claims_emitted == 0:
        return 0.0
    return min(1.0, max(0.0, claims_with_evidence / total_claims_emitted))


def honesty_weighted_pheromone(signal_value: float,
                               half_life_seconds: float,
                               claims_with_evidence: int,
                               total_claims_emitted: int,
                               timestamp: datetime = None) -> float:
    """Apply exponential decay and honesty weighting to a raw pheromone signal.

    Parameters
    ----------
    signal_value : float
        Raw magnitude of the pheromone.
    half_life_seconds : float
        Half‑life for exponential decay.
    claims_with_evidence : int
        Number of supported claims.
    total_claims_emitted : int
        Total number of emitted claims.
    timestamp : datetime, optional
        When the signal was emitted; defaults to ``datetime.now()``.

    Returns
    -------
    float
        Adjusted pheromone strength.
    """
    if timestamp is None:
        timestamp = datetime.now()
    now = datetime.now()
    elapsed = (now - timestamp).total_seconds()
    decay_factor = 0.5 ** (elapsed / half_life_seconds)

    honesty = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    return signal_value * decay_factor * honesty


# ----------------------------------------------------------------------
# Parent B – Clifford‑algebra geometric product and hybrid loss components
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a simplified geometric product of two vectors.

    The full Clifford algebra product is reduced to
    ``a * b + a ∧ b`` where ``∧`` is the exterior (wedge) product.
    For vectors this can be expressed as the outer product plus the inner product
    on the diagonal.

    Returns
    -------
    np.ndarray
        A matrix representing the multivector product.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Inner (dot) product on the diagonal
    inner = np.dot(a, b) * np.eye(a.shape[0])

    # Exterior (wedge) product as the antisymmetric part of the outer product
    outer = np.outer(a, b)
    wedge = outer - outer.T

    return inner + wedge


def ssim(x: np.ndarray, y: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """A very lightweight structural similarity index (SSIM) for 1‑D arrays.

    The implementation follows the classic formula but without windowing.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


def hybrid_loss(W: np.ndarray,
                x: np.ndarray,
                H: np.ndarray,
                rlct_lambda: float,
                honesty_factor: float,
                ssim_weight: float = 1.0) -> Tuple[float, Tuple[float, float, float]]:
    """Compute the unified hybrid loss.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix (d × d).
    x : np.ndarray
        Input vector (d,).
    H : np.ndarray
        Stylometry‑hash matrix (d × d).
    rlct_lambda : float
        RLCT estimate from Parent A.
    honesty_factor : float
        Honesty‑weighted pheromone scalar (≥0).
    ssim_weight : float, optional
        Optional scaling for the SSIM component.

    Returns
    -------
    loss : float
        Total hybrid loss.
    coeffs : tuple
        The three coefficients (α, β, γ) used.
    """
    # Coefficients derived from RLCT
    alpha = 1.0 / (1.0 + math.exp(-rlct_lambda))
    beta  = 1.0 / (1.0 + math.exp( rlct_lambda))
    gamma = abs(rlct_lambda) / (1.0 + abs(rlct_lambda))

    # Apply honesty factor to beta (strengthening the hash regularizer)
    beta *= honesty_factor

    # --- Component losses -------------------------------------------------
    Wx = W @ x
    L_TTT = np.linalg.norm(Wx - x) ** 2
    L_hash = np.linalg.norm(W - H, ord='fro') ** 2
    L_ssim = (1.0 - ssim(Wx, x)) * ssim_weight

    total = alpha * L_TTT + beta * L_hash + gamma * L_ssim
    return total, (alpha, beta, gamma)


def hybrid_update(W: np.ndarray,
                  x: np.ndarray,
                  H: np.ndarray,
                  rlct_lambda: float,
                  honesty_factor: float,
                  learning_rate: float = 0.01,
                  ssim_weight: float = 1.0) -> np.ndarray:
    """Perform a single gradient descent step on the hybrid loss.

    Returns the updated weight matrix.
    """
    # Coefficients (same as in hybrid_loss)
    alpha = 1.0 / (1.0 + math.exp(-rlct_lambda))
    beta  = 1.0 / (1.0 + math.exp( rlct_lambda))
    gamma = abs(rlct_lambda) / (1.0 + abs(rlct_lambda))
    beta *= honesty_factor

    # Gradients of each term
    Wx = W @ x
    grad_TTT = 2.0 * np.outer(Wx - x, x)                     # d×d
    grad_hash = 2.0 * (W - H)                                # d×d

    # Approximate gradient of SSIM w.r.t. W via chain rule on Wx
    # Using d(1-SSIM)/dWx ≈ -2*(Wx - x) / (||x||^2 + eps)
    eps = 1e-12
    denom = (np.linalg.norm(x) ** 2) + eps
    grad_ssim_Wx = -2.0 * (Wx - x) / denom
    grad_ssim = np.outer(grad_ssim_Wx, x)                    # d×d

    # Weighted sum
    grad_total = alpha * grad_TTT + beta * grad_hash + gamma * grad_ssim

    # Gradient descent update
    W_new = W - learning_rate * grad_total
    return W_new


# ----------------------------------------------------------------------
# Convenience class that bundles the pieces together
# ----------------------------------------------------------------------
class HybridFusion:
    """Encapsulates the full hybrid system.

    It holds the weight matrix, the stylometry hash, and provides methods to
    evaluate loss and update weights while internally handling RLCT estimation
    and pheromone honesty weighting.
    """

    def __init__(self,
                 dim: int,
                 half_life_seconds: float = 3600.0):
        self.dim = dim
        self.W = np.eye(dim) + 0.01 * np.random.randn(dim, dim)
        self.H = np.zeros((dim, dim))
        self.half_life_seconds = half_life_seconds
        self.pheromone_history: Dict[str, Tuple[float, datetime, int, int]] = {}

    def register_pheromone(self,
                           key: str,
                           raw_value: float,
                           claims_with_evidence: int,
                           total_claims_emitted: int,
                           timestamp: datetime = None) -> None:
        """Store a pheromone signal; later used as the honesty factor."""
        self.pheromone_history[key] = (raw_value,
                                       timestamp or datetime.now(),
                                       claims_with_evidence,
                                       total_claims_emitted)

    def _current_honesty_factor(self) -> float:
        """Aggregate all stored pheromones into a single scalar."""
        if not self.pheromone_history:
            return 1.0  # neutral factor
        factors = []
        for raw, ts, ev, tot in self.pheromone_history.values():
            adj = honesty_weighted_pheromone(
                raw,
                self.half_life_seconds,
                ev,
                tot,
                timestamp=ts)
            factors.append(adj)
        # Normalise to a reasonable range (0, 2)
        return max(0.0, min(2.0, np.mean(factors)))

    def train_step(self,
                   x: np.ndarray,
                   train_losses: list,
                   n_vals: list,
                   learning_rate: float = 0.01) -> float:
        """One hybrid training iteration.

        Returns the loss value computed before the update.
        """
        rlct = estimate_rlct(train_losses, n_vals)
        honesty = self._current_honesty_factor()
        loss, coeffs = hybrid_loss(self.W, x, self.H, rlct, honesty)
        self.W = hybrid_update(self.W, x, self.H,
                               rlct, honesty,
                               learning_rate=learning_rate)
        return loss

    def compute_geometric_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Expose the geometric product utility."""
        return geometric_product(a, b)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensionality
    d = 5
    fusion = HybridFusion(dim=d, half_life_seconds=1800.0)

    # Simulated pheromone signals
    fusion.register_pheromone(
        key="source_A",
        raw_value=1.2,
        claims_with_evidence=8,
        total_claims_emitted=10)

    fusion.register_pheromone(
        key="source_B",
        raw_value=0.7,
        claims_with_evidence=3,
        total_claims_emitted=5)

    # Dummy data for a single training step
    x_vec = np.random.randn(d)

    # Simulated loss curve (loss decreasing with larger n)
    n_vals = [10, 30, 100, 300, 1000]
    train_losses = [0.9, 0.6, 0.4, 0.25, 0.15]

    loss_before = fusion.train_step(x_vec, train_losses, n_vals, learning_rate=0.05)
    print(f"Hybrid loss before update: {loss_before:.6f}")

    # Verify geometric product works
    gp = fusion.compute_geometric_product(np.arange(d), np.arange(d, 0, -1))
    print("Geometric product matrix:\n", gp)

    # Final weight matrix sanity check
    print("Updated weight matrix W:\n", fusion.W)