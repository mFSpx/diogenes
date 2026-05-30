# DARWIN HAMMER — match 5619, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m1840_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s0.py (gen6)
# born: 2026-05-30T00:03:29Z

"""Hybrid NLMS‑Rectified‑Flow ↔ Physarum‑Pheromone Fusion

Parents:
- hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m1840_s1.py (NLMS + Gaussian‑RBF)
- hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s0.py (Physarum flux & pheromone decay)

Mathematical bridge:
The NLMS predictor consumes a feature vector φ that now includes the
Physarum conductance g as an additional scalar.  The prediction error ε drives
both the NLMS weight adaptation *and* the Physarum conductance update.
Conversely, the conductance influences the Gaussian‑RBF kernel amplitude via a
propensity term p = tanh(g), thereby closing a bidirectional loop between the
two topologies.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------
# Core utilities (from Parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised LMS weight update.

    Returns updated weights and the prediction error.
    """
    y = nlms_predict(weights, x)
    err = target - y
    norm = np.dot(x, x) + eps
    new_w = weights + (mu / norm) * err * x
    return new_w, err


def interpolant(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation Z_t = t·x1 + (1‑t)·x0."""
    return t * x1 + (1.0 - t) * x0


def gaussian_kernel(c: np.ndarray, z: np.ndarray, sigma: float) -> float:
    """RBF k(c, z) = exp(-‖c‑z‖²/(2σ²))."""
    diff = c - z
    return math.exp(-np.dot(diff, diff) / (2.0 * sigma * sigma))


# ----------------------------------------------------------------------
# Physarum‑Pheromone utilities (from Parent B)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_len: float, p_a: float, p_b: float, eps: float = 1e-12) -> float:
    """Flux q = g/ℓ * (p_a‑p_b)."""
    if edge_len <= 0:
        raise ValueError("edge_len must be positive")
    return conductance / max(edge_len, eps) * (p_a - p_b)


def update_conductance(
    conductance: float,
    q: float,
    reward: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    """
    Conductance dynamics with reward‑modulated growth.
    g_{t+1} = max(0, g + dt·(gain·(|q| + reward) - decay·g))
    """
    growth = gain * (abs(q) + reward)
    new_g = conductance + dt * (growth - decay * conductance)
    return max(0.0, new_g)


# ----------------------------------------------------------------------
# Hybrid state container
# ----------------------------------------------------------------------
@dataclass
class HybridState:
    weights: np.ndarray          # NLMS weight vector
    conductance: float           # Physarum edge conductance
    pressure_a: float            # Pressure at node A
    pressure_b: float            # Pressure at node B
    sigma: float                 # RBF bandwidth
    edge_len: float = 1.0        # Physical length of the edge (fixed)


# ----------------------------------------------------------------------
# Hybrid step – integrates both parent dynamics
# ----------------------------------------------------------------------
def hybrid_step(
    state: HybridState,
    x0: np.ndarray,
    x1: np.ndarray,
    t: float,
    target: float,
    context: np.ndarray,
    mu: float = 0.5,
) -> Tuple[HybridState, float]:
    """
    Perform one hybrid iteration:
    1. Interpolate flow state Z_t.
    2. Compute propensity p = tanh(g) and modulate the RBF amplitude.
    3. Build feature vector φ = [Z_t , p·k(c, Z_t) , g].
    4. NLMS predict & update using φ.
    5. Use prediction error as reward to update conductance via flux.
    Returns the updated state and the scalar prediction.
    """
    # 1. Interpolation
    Z = interpolant(x0, x1, t)

    # 2. Propensity from conductance
    propensity = math.tanh(state.conductance)

    # 3. Kernel feature (modulated by propensity)
    k_val = gaussian_kernel(context, Z, state.sigma)
    kernel_feat = propensity * k_val

    # Assemble feature vector φ
    phi = np.concatenate([Z, np.array([kernel_feat, state.conductance])])

    # 4. NLMS prediction & weight update
    new_weights, err = nlms_update(state.weights, phi, target, mu=mu)

    # 5. Flux computation (using current pressures)
    q = flux(state.conductance, state.edge_len, state.pressure_a, state.pressure_b)

    # Reward is higher when error magnitude is low (negative absolute error)
    reward = -abs(err)

    # Conductance update with reward‑modulated growth
    new_conductance = update_conductance(state.conductance, q, reward)

    # Optional: simple pressure relaxation (pheromone‑like decay)
    new_pressure_a = state.pressure_a * (1.0 - 0.01) + 0.01 * state.pressure_b
    new_pressure_b = state.pressure_b * (1.0 - 0.01) + 0.01 * state.pressure_a

    # Pack updated state
    new_state = HybridState(
        weights=new_weights,
        conductance=new_conductance,
        pressure_a=new_pressure_a,
        pressure_b=new_pressure_b,
        sigma=state.sigma,
        edge_len=state.edge_len,
    )

    prediction = nlms_predict(new_weights, phi)
    return new_state, prediction


# ----------------------------------------------------------------------
# Additional helper demonstrating batch processing (third required function)
# ----------------------------------------------------------------------
def run_hybrid_episode(
    init_state: HybridState,
    xs0: np.ndarray,
    xs1: np.ndarray,
    ts: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray,
    mu: float = 0.5,
) -> Tuple[HybridState, np.ndarray]:
    """
    Executes a sequence of hybrid steps over aligned arrays.
    Returns the final state and the array of predictions.
    """
    state = init_state
    preds = []
    for x0, x1, t, y, ctx in zip(xs0, xs1, ts, targets, contexts):
        state, pred = hybrid_step(state, x0, x1, t, y, ctx, mu=mu)
        preds.append(pred)
    return state, np.array(preds)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    dim_flow = 4
    # Random flow endpoints
    x0 = np.random.randn(dim_flow)
    x1 = np.random.randn(dim_flow)

    # Context vector (same dimension for simplicity)
    context = np.random.randn(dim_flow)

    # Initialise hybrid state
    init_weights = np.random.randn(dim_flow + 2)  # Z + kernel_feat + conductance
    init_state = HybridState(
        weights=init_weights,
        conductance=1.0,
        pressure_a=1.0,
        pressure_b=0.5,
        sigma=1.0,
    )

    # Single step demonstration
    t = 0.3
    target = float(np.sum(interpolant(x0, x1, t)))  # synthetic target
    state, pred = hybrid_step(init_state, x0, x1, t, target, context)
    print(f"Prediction after one step: {pred:.4f}, error: {target - pred:.4f}")

    # Run a short episode
    steps = 10
    xs0 = np.tile(x0, (steps, 1))
    xs1 = np.tile(x1, (steps, 1))
    ts = np.linspace(0, 1, steps)
    targets = np.array([float(np.sum(interpolant(x0, x1, ti))) for ti in ts])
    contexts = np.tile(context, (steps, 1))

    final_state, predictions = run_hybrid_episode(
        init_state, xs0, xs1, ts, targets, contexts, mu=0.4
    )
    print("Predictions over episode:", predictions.round(4))
    print("Final conductance:", final_state.conductance)
    print("Final weights norm:", np.linalg.norm(final_state.weights))