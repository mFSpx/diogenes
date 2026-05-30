# DARWIN HAMMER — match 1308, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s1.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_fold_change_detectio_m32_s1.py (gen2)
# born: 2026-05-29T23:35:05Z

"""HybridPathKANFoldChange

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – Path‑signature / Kolmogorov‑Arnold Network (KAN) side.
  It provides a *lead‑lag transform* that encodes causality and a *B‑spline basis*
  that approximates the iterated‑integral signature using the same spline
  functions that KANs employ as activation maps.

* **Parent B** – Fold‑change detection / VRAM‑scheduler side.
  It supplies a *fold‑change detector* (ratio of successive state magnitudes) and
  a *gradient‑descent‑with‑Euler* update for a weight matrix **W**.

The mathematical bridge is the **weight matrix W** that maps the spline‑expanded
features (the “signature vector”) to an output.  In the hybrid we let the
gradient‑descent step be *modulated* by the fold‑change detector: the effective
learning‑rate is multiplied by a factor `γ = 1 + κ·Δ`, where `Δ` is the relative
fold‑change of the loss and `κ` is a tunable gain.  The matrix update is performed
with a single Euler integration step, preserving the continuous‑time flavour of
the fold‑change algorithm while keeping the discrete‑gradient nature of the
KAN‑style trainer.

The module therefore contains:

1. `lead_lag_transform` – unchanged from Parent A.
2. `bspline_basis` – unchanged from Parent A.
3. `fold_change_factor` – computes the fold‑change factor from two scalar
   quantities.
4. `euler_weight_update` – performs an Euler integration of the weight matrix
   using the modulated learning‑rate.
5. `hybrid_step` – a single training step that ties the three pieces together:
   transform a path, expand it with B‑splines, compute a simple squared‑error
   loss, obtain its gradient w.r.t. **W**, and update **W** with the fold‑change
   aware Euler rule.

The implementation uses only the allowed standard‑library modules and NumPy."""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Tuple

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dim)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of order `k` at positions `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = x.shape[0]
    B = np.zeros((N, len(t) - 1), dtype=np.float64)

    # Zeroth‑order basis (piecewise constant)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recursion for higher orders
    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else np.zeros(N)

            B_new[:, i] = term_l + term_r
        B = B_new

    return B

# ----------------------------------------------------------------------
# Parent B building blocks (adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def fold_change_factor(prev: float, curr: float, gain: float = 0.5) -> float:
    """
    Compute a multiplicative factor based on the relative fold‑change
    between two scalar quantities.
    γ = 1 + gain * (curr/prev - 1)
    Guard against division by zero.
    """
    if prev == 0.0:
        return 1.0
    delta = curr / prev - 1.0
    return 1.0 + gain * delta

def euler_weight_update(W: np.ndarray,
                       grad: np.ndarray,
                       lr: float,
                       dt: float,
                       mod_factor: float) -> np.ndarray:
    """
    Perform an Euler integration step for the weight matrix.
    dW/dt = - lr * grad * mod_factor
    W_new = W + dt * dW/dt
    """
    dW = -lr * grad * mod_factor
    return W + dt * dW

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_step(path: np.ndarray,
                W: np.ndarray,
                target: np.ndarray,
                lr: float = 0.01,
                dt: float = 1.0,
                gain: float = 0.5,
                spline_grid_size: int = 10) -> Tuple[np.ndarray, float]:
    """
    Execute one hybrid training step.

    1. Lead‑lag transform the raw path.
    2. For each transformed dimension, evaluate a B‑spline basis on a uniform grid.
       The concatenated basis matrix forms the feature vector Φ.
    3. Compute a linear prediction `y = Φ @ W`.
    4. Compute squared‑error loss `L = ½‖y - target‖²`.
    5. Gradient w.r.t. W is `∂L/∂W = Φᵀ (y - target)`.
    6. Compute the fold‑change factor from the previous loss (cached on W via
       an attribute) to the current loss and modulate the learning‑rate.
    7. Update W with an Euler step.

    The function returns the updated weight matrix and the current loss.
    """
    # ----- 1. Lead‑lag -----
    transformed = lead_lag_transform(path)          # shape (2T‑1, 2d)

    # ----- 2. B‑spline expansion -----
    # Build a uniform grid spanning the observed values per dimension.
    mins = transformed.min(axis=0)
    maxs = transformed.max(axis=0)
    # Avoid degenerate grid when min == max
    eps = 1e-8
    grid = np.linspace(mins, maxs + eps, spline_grid_size, axis=0)  # shape (G, dim)

    # Evaluate basis for each dimension independently and concatenate.
    basis_list = []
    for dim in range(transformed.shape[1]):
        x = transformed[:, dim]
        grid_dim = grid[:, dim]
        B = bspline_basis(x, grid_dim, k=3)          # (N, G)
        basis_list.append(B)
    Phi = np.concatenate(basis_list, axis=1)        # (N, G * dim)

    # ----- 3. Linear prediction -----
    # Ensure W matches the feature dimension
    if W.shape[0] != Phi.shape[1]:
        raise ValueError(f"Weight matrix shape {W.shape} incompatible with feature size {Phi.shape[1]}")
    y = Phi @ W                                      # (N, out_dim)

    # ----- 4. Loss -----
    diff = y - target
    loss = 0.5 * np.mean(diff ** 2)

    # ----- 5. Gradient -----
    grad = (Phi.T @ diff) / diff.shape[0]           # (features, out_dim)

    # ----- 6. Fold‑change modulation -----
    # Store previous loss inside the weight matrix via a private attribute.
    prev_loss = getattr(W, "_prev_loss", loss)
    mod = fold_change_factor(prev_loss, loss, gain=gain)

    # ----- 7. Euler update -----
    W_new = euler_weight_update(W, grad, lr, dt, mod)

    # Attach the current loss for the next call.
    setattr(W_new, "_prev_loss", loss)

    return W_new, loss

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic path: 5 time steps, 2‑D coordinates
    np.random.seed(42)
    path = np.cumsum(np.random.randn(5, 2), axis=0)

    # Target: we want the model to output a constant vector (e.g., zeros)
    # after spline expansion we will have N rows; we pick the mean of Φ as target.
    transformed = lead_lag_transform(path)
    mins = transformed.min(axis=0)
    maxs = transformed.max(axis=0)
    grid = np.linspace(mins, maxs + 1e-8, 8, axis=0)
    basis_list = [bspline_basis(transformed[:, d], grid[:, d]) for d in range(transformed.shape[1])]
    Phi = np.concatenate(basis_list, axis=1)

    target = np.zeros_like(Phi @ np.zeros((Phi.shape[1], 1)))  # shape (N, 1)

    # Initialise weight matrix with small random values.
    W = np.random.randn(Phi.shape[1], target.shape[1]) * 0.01

    # Run a few hybrid steps.
    for epoch in range(10):
        W, loss = hybrid_step(path, W, target, lr=0.05, dt=1.0, gain=0.3, spline_grid_size=8)
        print(f"Epoch {epoch+1:02d} – loss: {loss:.6f}")

    # Demonstrate that VramSlotPlan can still be instantiated (unused in the hybrid)
    plan = VramSlotPlan(
        artifact_id="model_xyz",
        artifact_kind="lora",
        action="preempt",
        estimated_mb=512,
        reason="memory pressure",
        detail={"timestamp": "2026-05-29T23:30:00Z"}
    )
    print("Sample VramSlotPlan:", plan.as_dict())