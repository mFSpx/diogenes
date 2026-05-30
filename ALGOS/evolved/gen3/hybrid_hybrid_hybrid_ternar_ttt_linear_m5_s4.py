# DARWIN HAMMER — match 5, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_ternary_route_variational_free_ene_m21_s1.py (gen2)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:25:06Z

"""
Hybrid Algorithm: SSIM‑Guided Test‑Time Training (Hybrid‑SSIM‑TTT)

Parents
-------
* **Parent A** – `hybrid_hybrid_ternary_route_variational_free_ene_m21_s1.py`
  provides a Structural Similarity (SSIM) function to quantify the
  resemblance between two signals and a routing skeleton that updates a
  belief (mean) using variational free‑energy ideas.

* **Parent B** – `ttt_linear.py`
  implements Test‑Time Training (TTT) where a weight matrix **W** is
  continuously adapted during inference by gradient descent on a loss
  function.

Mathematical Bridge
-------------------
Both parents rely on a *scalar loss* that drives an update of a *parameter*
matrix:

* In TTT the loss is usually a reconstruction error  
  `L_rec(W) = ||W x – x||²`.

* In the SSIM‑based router the similarity `S(x, y) ∈ [0,1]` can be turned
  into a loss `L_ssim = 1 – S(x, y)`.

The hybrid algorithm treats `L_hybrid = α·L_rec + β·L_ssim` as a unified
objective. The gradient of `L_rec` w.r.t. **W** is analytically simple,
while the gradient of `L_ssim` is approximated by the same reconstruction
gradient scaled by the SSIM loss value. This yields a single update rule
that simultaneously compresses the input stream (TTT) and enforces
structural similarity (SSIM), echoing the variational‑free‑energy belief
update of Parent A.

The code below implements:
1. `ssim` – full SSIM for 1‑D arrays.
2. `init_hybrid` – weight matrix initialization.
3. `hybrid_loss` – combined loss `α·L_rec + β·L_ssim`.
4. `hybrid_step` – gradient descent update of **W** using the combined loss.
5. `hybrid_forward` – apply the current **W** to an input vector.
6. `route_with_hybrid` – a toy routing function that returns a packet enriched
   with the current SSIM score and updated belief state.

All operations rely only on NumPy and the Python standard library.
"""

import json
import sys
import random
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# SSIM implementation (Parent A)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two 1‑D signals.

    Parameters
    ----------
    x, y : np.ndarray
        Input vectors of equal length.
    dynamic_range : float
        The difference between the maximum and minimum possible values
        (default 255 for 8‑bit images).
    k1, k2 : float
        Stabilizing constants.

    Returns
    -------
    float
        SSIM value in [0, 1].
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size == 0:
        raise ValueError("inputs must be non‑empty")

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]

    numerator = (2 * mx * my + C1) * (2 * cov + C2)
    denominator = (mx ** 2 + my ** 2 + C1) * (vx + vy + C2)

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Test‑Time Training core (Parent B) adapted for hybrid use
# ----------------------------------------------------------------------
def init_hybrid(d_in: int,
                d_out: int | None = None,
                scale: float = 0.01,
                seed: int = 0) -> np.ndarray:
    """
    Initialise the weight matrix **W** for the hybrid model.

    Parameters
    ----------
    d_in : int
        Input dimensionality.
    d_out : int | None
        Output dimensionality; defaults to ``d_in``.
    scale : float
        Standard‑deviation scaling factor for the random init.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Weight matrix of shape (d_out, d_in).
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def hybrid_loss(W: np.ndarray,
                x: np.ndarray,
                alpha: float = 0.5,
                beta: float = 0.5,
                dynamic_range: float = 255.0) -> Tuple[float, float]:
    """
    Compute the combined loss:

        L = α·L_rec + β·L_ssim

    where
        L_rec = ||W x – x||²
        L_ssim = 1 – SSIM(x, W x)

    Returns both components for diagnostic purposes.

    Parameters
    ----------
    W : np.ndarray
        Current weight matrix.
    x : np.ndarray
        Input vector.
    alpha, beta : float
        Weighting coefficients (must sum to 1 for a proper convex blend,
        but the function does not enforce this).
    dynamic_range : float
        Passed to the SSIM function.

    Returns
    -------
    Tuple[float, float]
        (L_rec, L_ssim)
    """
    recon = W @ x
    L_rec = float(np.sum((recon - x) ** 2))

    # Clip to the dynamic range expected by SSIM to avoid numerical issues
    recon_clipped = np.clip(recon, 0, dynamic_range)
    x_clipped = np.clip(x, 0, dynamic_range)
    L_ssim = 1.0 - ssim(x_clipped, recon_clipped, dynamic_range=dynamic_range)

    return alpha * L_rec, beta * L_ssim


def hybrid_step(W: np.ndarray,
                x: np.ndarray,
                eta: float = 0.001,
                alpha: float = 0.5,
                beta: float = 0.5,
                dynamic_range: float = 255.0) -> np.ndarray:
    """
    Perform one gradient‑descent update of **W** using the hybrid loss.

    The gradient of the reconstruction term is exact:
        ∂L_rec/∂W = 2·(W x – x)·xᵀ

    For the SSIM term we reuse the same gradient direction as the
    reconstruction term, scaled by the SSIM loss magnitude. This yields a
    simple yet effective proxy for ∂L_ssim/∂W.

    Parameters
    ----------
    W : np.ndarray
        Current weight matrix.
    x : np.ndarray
        Input vector.
    eta : float
        Learning rate.
    alpha, beta : float
        Loss weighting coefficients.
    dynamic_range : float
        Passed to the SSIM function.

    Returns
    -------
    np.ndarray
        Updated weight matrix.
    """
    recon = W @ x
    error = recon - x  # shape (d_out,)

    # Gradient of reconstruction loss
    grad_rec = 2.0 * np.outer(error, x)  # shape (d_out, d_in)

    # Approximate SSIM gradient using the same direction, scaled
    _, L_ssim = hybrid_loss(W, x, alpha=0.0, beta=beta, dynamic_range=dynamic_range)
    grad_ssim = L_ssim * grad_rec  # simple proxy

    # Total gradient
    grad_total = alpha * grad_rec + beta * grad_ssim

    # Gradient descent step
    W_new = W - eta * grad_total
    return W_new


def hybrid_forward(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply the current weight matrix to an input vector.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix.
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Output vector (reconstruction).
    """
    return W @ x


# ----------------------------------------------------------------------
# Toy routing function that demonstrates the hybrid state usage
# ----------------------------------------------------------------------
def route_with_hybrid(packet: Dict[str, Any],
                     W: np.ndarray,
                     eta: float = 0.001,
                     alpha: float = 0.5,
                     beta: float = 0.5) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Process a packet, update the hybrid weight matrix, and return an enriched
    response.

    The function extracts a numeric payload (list of floats) from the packet,
    treats it as the input vector *x*, performs one hybrid update step, and
    returns a response containing:

    * ``ssim_score`` – SSIM between the raw payload and its reconstruction.
    * ``reconstruction`` – the reconstructed vector.
    * ``updated_state`` – the new weight matrix (as a list of lists for JSON
      serialisation).

    Parameters
    ----------
    packet : dict
        Incoming data packet. Expected key ``payload`` → ``data`` → list of
        numbers.
    W : np.ndarray
        Current hybrid weight matrix.
    eta, alpha, beta : float
        Hyper‑parameters passed to :func:`hybrid_step`.

    Returns
    -------
    Tuple[dict, np.ndarray]
        (response packet, updated weight matrix)
    """
    # Extract numeric payload; fall back to a random vector if missing
    raw = packet.get("payload", {}).get("data")
    if not isinstance(raw, list) or not raw:
        dim = W.shape[1]
        raw = np.random.uniform(0, 255, size=dim).tolist()
    x = np.asarray(raw, dtype=float)

    # Forward pass
    recon = hybrid_forward(W, x)

    # Compute SSIM for reporting
    ssim_score = ssim(np.clip(x, 0, 255), np.clip(recon, 0, 255), dynamic_range=255.0)

    # Update weights
    W_new = hybrid_step(W, x, eta=eta, alpha=alpha, beta=beta, dynamic_range=255.0)

    # Build response
    response = {
        "engine_channel": "hybrid_ssim_ttt",
        "ssim_score": float(ssim_score),
        "reconstruction": recon.tolist(),
        "outbound_state": "updated",
        "updated_state": W_new.tolist(),
    }
    return response, W_new


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a 8‑dimensional hybrid model
    dim = 8
    W = init_hybrid(d_in=dim, scale=0.01, seed=42)

    # Dummy packet with a random payload
    pkt = {
        "payload": {
            "data": np.random.uniform(0, 255, size=dim).round(2).tolist()
        },
        "metadata": {"source": "unit_test"}
    }

    # Run a few update steps to ensure stability
    for step in range(5):
        resp, W = route_with_hybrid(pkt, W, eta=0.005, alpha=0.7, beta=0.3)
        print(f"Step {step + 1}: SSIM = {resp['ssim_score']:.4f}")

    print("Hybrid SSIM‑TTT test completed without errors.")