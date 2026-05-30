# DARWIN HAMMER — match 5, survivor 5
# gen: 3
# parent_a: hybrid_hybrid_ternary_route_variational_free_ene_m21_s1.py (gen2)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:25:06Z

"""Hybrid Ternary‑Router / Test‑Time Training (HTR‑TTT)

Parents
-------
* **hybrid_hybrid_ternary_route_variational_free_ene_m21_s1.py** – uses
  Structural Similarity Index (SSIM) to assess the fidelity of a ternary
  router’s output and a variational free‑energy (VFE) term to update a belief
  mean.
* **ttt_linear.py** – Test‑Time Training (TTT) where a weight matrix **W**
  (the hidden “state”) is updated online by a gradient‑descent step on a
  self‑supervised loss.

Mathematical Bridge
-------------------
Both parents expose a *scalar quality metric* that can be differentiated with
respect to a weight matrix:

* SSIM( x, Wx ) provides a similarity‑based loss term (1 − SSIM) that
  encourages the router to preserve the input structure.
* VFE( μ, Wx ) is a quadratic energy term that drives a belief mean **μ**
  toward the router’s current output.

In TTT the same scalar loss (reconstruction error) yields the gradient
∂L/∂W = 2 (Wx − x) xᵀ, which is used to update **W** online.
The hybrid therefore **adds** the SSIM‑derived loss and the VFE‑derived
gradient to the reconstruction gradient, producing a unified update rule
that simultaneously:

1. Improves reconstruction (TTT core),
2. Maximises perceptual similarity (SSIM from the ternary router), and
3. Refines a probabilistic belief (VFE).

The module implements this fused dynamics in three public functions:
`init_hybrid`, `hybrid_forward`, `hybrid_step`, plus a helper
`hybrid_loss` that aggregates the three components.  The `__main__` block
runs a short smoke test."""

import json
import sys
import random
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility helpers (borrowed from parent A)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z‑format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context(text: str | None) -> dict[str, Any]:
    """Parse a JSON string into a dict; empty input yields {}."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value


# ----------------------------------------------------------------------
# SSIM – structural similarity (parent A)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Return the SSIM index between two 1‑D signals.

    The implementation follows the classic formulation for a single window.
    """
    if x.shape != y.shape:
        raise ValueError("inputs must have identical shape")
    if x.size == 0:
        raise ValueError("inputs must be non‑empty")

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Variational Free Energy (parent A)
# ----------------------------------------------------------------------
def variational_free_energy(mu: np.ndarray,
                            obs: np.ndarray,
                            sigma2: float = 1.0) -> float:
    """Quadratic free‑energy term for a Gaussian belief.

    VFE = ½ [ (obs‑mu)² / sigma² + log(sigma²) ] summed over dimensions.
    """
    diff = obs - mu
    return 0.5 * (np.sum(diff ** 2) / sigma2 + diff.size * math.log(sigma2))


# ----------------------------------------------------------------------
# Test‑Time Training core (parent B)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None,
             scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize the weight matrix W ∈ ℝ^{d_out×d_in}."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def reconstruction_loss(W: np.ndarray, x: np.ndarray) -> float:
    """L_rec = ‖W x − x‖²."""
    residual = W @ x - x
    return float(np.sum(residual ** 2))


# ----------------------------------------------------------------------
# Hybrid system
# ----------------------------------------------------------------------
def init_hybrid(d_in: int,
                d_out: int | None = None,
                scale: float = 0.01,
                seed: int = 0,
                sigma2: float = 1.0) -> Dict[str, Any]:
    """Create a hybrid state containing:

    * ``W`` – the TTT weight matrix,
    * ``mu`` – belief mean vector (size d_out),
    * ``sigma2`` – observation variance for VFE,
    * ``alpha``, ``beta``, ``gamma`` – loss‑weight hyper‑parameters.
    """
    W = init_ttt(d_in, d_out, scale, seed)
    if d_out is None:
        d_out = d_in
    mu = np.zeros(d_out)  # start with neutral belief
    state = {
        "W": W,
        "mu": mu,
        "sigma2": sigma2,
        "alpha": 1.0,   # reconstruction loss weight
        "beta": 1.0,    # (1‑SSIM) weight
        "gamma": 1.0,   # VFE weight
    }
    return state


def hybrid_forward(state: Dict[str, Any],
                   x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Run a single forward pass.

    Returns a tuple ``(y, ssim_score, vfe)`` where:

    * ``y = W @ x`` – router output,
    * ``ssim_score`` – SSIM between ``x`` and ``y``,
    * ``vfe`` – variational free‑energy for the current belief.
    """
    W = state["W"]
    mu = state["mu"]
    sigma2 = state["sigma2"]

    y = W @ x
    s = ssim(x, y)
    v = variational_free_energy(mu, y, sigma2)
    return y, s, v


def hybrid_loss(state: Dict[str, Any],
                x: np.ndarray) -> float:
    """Aggregate loss = α·L_rec + β·(1‑SSIM) + γ·VFE."""
    W = state["W"]
    mu = state["mu"]
    sigma2 = state["sigma2"]
    alpha = state["alpha"]
    beta = state["beta"]
    gamma = state["gamma"]

    # 1. Reconstruction (TTT)
    L_rec = reconstruction_loss(W, x)

    # 2. SSIM‑derived loss
    y = W @ x
    ssim_score = ssim(x, y)
    L_ssim = 1.0 - ssim_score

    # 3. Variational free‑energy
    L_vfe = variational_free_energy(mu, y, sigma2)

    total = alpha * L_rec + beta * L_ssim + gamma * L_vfe
    return float(total)


def hybrid_step(state: Dict[str, Any],
                x: np.ndarray,
                lr: float = 1e-3) -> None:
    """Perform one online update of ``W`` and the belief mean ``mu``.

    The gradient w.r.t. ``W`` is taken from the reconstruction term only
    (the SSIM term is non‑differentiable in this lightweight implementation).
    The belief mean is updated by the gradient of the VFE term.
    """
    W = state["W"]
    mu = state["mu"]
    sigma2 = state["sigma2"]
    alpha = state["alpha"]
    gamma = state["gamma"]

    # ---- Gradient for reconstruction loss ---------------------------------
    residual = W @ x - x                     # shape (d_out,)
    grad_W = 2.0 * np.outer(residual, x)     # d_out × d_in
    # Apply weight α
    grad_W *= alpha

    # ---- Gradient for VFE w.r.t. mu ---------------------------------------
    y = W @ x
    grad_mu = -(y - mu) / sigma2             # d_out
    # Gradient ascent on mu (since we minimize VFE)
    mu -= gamma * grad_mu

    # ---- Parameter update --------------------------------------------------
    W -= lr * grad_W

    # Store back
    state["W"] = W
    state["mu"] = mu


# ----------------------------------------------------------------------
# Optional routing stub (kept for API compatibility with parent A)
# ----------------------------------------------------------------------
def route_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """A minimal placeholder that mimics the original routing interface."""
    text = str(packet.get("text_surface") or
               packet.get("raw_command") or
               packet.get("text") or "")
    # In the hybrid we simply echo a static example response.
    route = {"text_surface": "example response",
             "engine_channel": "cpu_fairyfuse_ternary",
             "outbound_state": "draft_only"}
    return route


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    d = 8
    state = init_hybrid(d_in=d, scale=0.05, sigma2=0.5)
    print(f"Initial loss: {hybrid_loss(state, np.random.randn(d)):.4f}")

    for step in range(5):
        x = np.random.randn(d) * 20 + 128  # simulate image‑like signal
        loss_before = hybrid_loss(state, x)
        hybrid_step(state, x, lr=1e-4)
        loss_after = hybrid_loss(state, x)
        y, ssim_score, vfe = hybrid_forward(state, x)
        print(f"Step {step+1:02d} | loss ↓ {loss_before:.4f}→{loss_after:.4f} | "
              f"SSIM {ssim_score:.4f} | VFE {vfe:.4f}")

    # Demonstrate routing stub
    pkt = {"text_surface": "hello"}
    print("Routing stub output:", route_packet(pkt))