# DARWIN HAMMER — match 1861, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m886_s2.py (gen5)
# born: 2026-05-29T23:39:15Z

"""Hybrid Algorithm: Fusion of Simulated Annealing Leader Election (Parent A) and
TTT‑Linear Weight Matrix with Reconstruction‑Risk (Parent B).

Mathematical Bridge
-------------------
* Parent A provides a *broadcast probability* `p` (interpreted as a pressure field)
  and an exponential *cooling temperature* `T`.  
* Parent B updates a weight matrix `W` by gradient descent with a learning rate
  `η`.  

The fusion treats `p·T` as a *joint temperature* that simultaneously scales
the annealing schedule of the leader‑election process and the learning rate of
the TTT‑Linear update.  The pressure difference `Δp = pressure_a – pressure_b`
drives a *flux* `Φ = g·Δp / ℓ` (conductance `g`, edge length `ℓ`) which is used as
an additional regularisation term on `W`.  Thus the two dynamical systems are
coupled through the shared scalar fields `p`, `T`, and `Φ`.

The implementation below provides a self‑contained hybrid system with three core
functions demonstrating this integration.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

Node = int
Graph = Dict[Node, List[Node]]


@dataclass
class HybridState:
    # Leader‑election / annealing parameters (Parent A)
    phases: int
    phase: int
    t0: float = 1.0
    alpha: float = 0.95

    # Physarum‑like conductance parameters (Parent A)
    conductance: float = 1.0
    edge_length: float = 1.0
    pressure_a: float = 0.0
    pressure_b: float = 0.0

    # TTT‑Linear weight matrix (Parent B)
    W: np.ndarray | None = None  # shape (d_out, d_in)


# ----------------------------------------------------------------------
# Parent A – broadcast probability and cooling schedule
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0.0 <= alpha <= 1.0):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hybrid_temperature(state: HybridState) -> float:
    """Joint temperature = cooling * broadcast probability."""
    p = broadcast_probability(state.phases, state.phase)
    T = cooling_temperature(state.phase - 1, state.t0, state.alpha)
    return T * p


# ----------------------------------------------------------------------
# Parent B – TTT‑Linear weight matrix utilities
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix W ~ N(0, scale²)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean‑squared error."""
    return float(np.mean((pred - target) ** 2))


def reconstruction_risk(state: HybridState, X: np.ndarray, target: np.ndarray) -> float:
    """
    Risk = loss / (variance of target + ε).
    Provides a scalar that quantifies how well the current weights reconstruct the target.
    """
    if state.W is None:
        raise ValueError("Weight matrix W is not initialized")
    pred = X @ state.W.T  # shape (n_samples, d_out)
    loss = mse_loss(pred, target)
    var = float(np.var(target))
    return loss / (var + 1e-8)


# ----------------------------------------------------------------------
# Hybrid core dynamics
# ----------------------------------------------------------------------
def compute_flux(state: HybridState) -> float:
    """
    Physarum‑like flux Φ = g·Δp / ℓ.
    Δp is the pressure difference derived from broadcast probability and cooling temperature.
    """
    delta_p = state.pressure_a - state.pressure_b
    if state.edge_length == 0:
        return 0.0
    return state.conductance * delta_p / state.edge_length


def update_weights(state: HybridState, X: np.ndarray, target: np.ndarray) -> None:
    """
    Gradient descent on MSE with learning rate = hybrid_temperature.
    An additional regularisation proportional to the flux dampens the weights.
    """
    if state.W is None:
        raise ValueError("Weight matrix W is not initialized")
    # Forward pass
    pred = X @ state.W.T  # (n, d_out)
    # Gradient of MSE w.r.t. W
    grad = (2.0 / X.shape[0]) * (pred - target).T @ X  # (d_out, d_in)

    # Learning rate from joint temperature
    eta = hybrid_temperature(state)

    # Flux‑based regularisation term
    phi = compute_flux(state)
    reg = phi * 0.01 * state.W

    # Weight update
    state.W -= eta * (grad + reg)


def hybrid_step(state: HybridState, X: np.ndarray, target: np.ndarray) -> None:
    """
    Perform one iteration of the hybrid system:
      1. Update pressure fields from current phase.
      2. Compute flux.
      3. Update conductance (simple positive feedback).
      4. Update weight matrix using temperature‑scaled gradient descent.
      5. Advance phase counter.
    """
    # 1. Pressure fields
    state.pressure_a = broadcast_probability(state.phases, state.phase)
    state.pressure_b = cooling_temperature(state.phase - 1, state.t0, state.alpha)

    # 2. Flux (used later as regularisation)
    _ = compute_flux(state)

    # 3. Conductance adaptation (increase if pressure_a > pressure_b)
    delta = state.pressure_a - state.pressure_b
    state.conductance = max(0.1, state.conductance + 0.01 * delta)

    # 4. Weight update
    update_weights(state, X, target)

    # 5. Advance phase (wrap around after reaching phases)
    state.phase = (state.phase % state.phases) + 1


# ----------------------------------------------------------------------
# Helper to create a fresh HybridState with random weights
# ----------------------------------------------------------------------
def init_hybrid_state(
    phases: int,
    d_in: int,
    d_out: int | None = None,
    conductance: float = 1.0,
    edge_length: float = 1.0,
    t0: float = 1.0,
    alpha: float = 0.95,
    seed: int = 0,
) -> HybridState:
    """Factory that builds a fully‑initialised HybridState."""
    if d_out is None:
        d_out = d_in
    W = init_ttt(d_in, d_out, scale=0.01, seed=seed)
    return HybridState(
        phases=phases,
        phase=1,
        t0=t0,
        alpha=alpha,
        conductance=conductance,
        edge_length=edge_length,
        pressure_a=0.0,
        pressure_b=0.0,
        W=W,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic regression problem
    rng = np.random.default_rng(42)
    n_samples = 200
    d_in = 5
    d_out = 3

    X = rng.normal(size=(n_samples, d_in))
    true_W = rng.normal(size=(d_out, d_in))
    noise = rng.normal(scale=0.1, size=(n_samples, d_out))
    target = X @ true_W.T + noise

    # Initialise hybrid system
    state = init_hybrid_state(
        phases=10,
        d_in=d_in,
        d_out=d_out,
        conductance=1.0,
        edge_length=1.0,
        t0=1.0,
        alpha=0.95,
        seed=7,
    )

    # Run a few hybrid iterations
    for _ in range(30):
        hybrid_step(state, X, target)

    # Final diagnostics
    final_risk = reconstruction_risk(state, X, target)
    print(f"Final reconstruction risk: {final_risk:.6f}")
    print(f"Final conductance: {state.conductance:.4f}")
    print(f"Final temperature: {hybrid_temperature(state):.6f}")