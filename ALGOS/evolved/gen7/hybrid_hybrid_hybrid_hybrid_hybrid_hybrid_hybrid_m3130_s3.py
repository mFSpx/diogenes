# DARWIN HAMMER — match 3130, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1407_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_caputo_hybrid_nlms_omni_cha_m725_s1.py (gen4)
# born: 2026-05-29T23:47:59Z

"""
Hybrid Fusion of:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1407_s1.py (Morphology‑based regret utility)
- hybrid_hybrid_hybrid_caputo_hybrid_nlms_omni_cha_m725_s1.py (Caputo‑fractional NLMS)

Mathematical Bridge
-------------------
The regret‑weighted utility **U** computed from a lead‑lag transformed path
signature (parent A) is used as a time‑varying scaling factor for the
learning rate of a Normalized Least‑Mean‑Squares (NLMS) adaptive filter.
The evolution of the NLMS weight vector **w(t)** is governed by a Caputo
fractional derivative of order **α** (parent B).  Concretely

    μ̃(t) = μ * U(t)                     # scaled step size
    Dᶜα w(t) = (1/Γ(1‑α)) Σ_{τ=0}^{t‑1}
                (w(τ+1)‑w(τ)) / (t‑τ)^{α}

and the discrete NLMS update is applied to the fractional derivative
approximation, yielding a hybrid weight dynamics that simultaneously
captures morphological regret information and fractional memory effects.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class MathAction:
    id: str
    tokens: Tuple[str, ...]
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

# ----------------------------------------------------------------------
# Helper utilities (from Parent A)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity, dimensionless."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio, dimensionless."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return height / max(length, width)

def compute_morphology_features(morph: Morphology) -> np.ndarray:
    """Return a feature vector [sphericity, flatness, mass]."""
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flt = flatness_index(morph.length, morph.width, morph.height)
    return np.array([sph, flt, morph.mass], dtype=np.float64)

def compute_path_signature(actions: List[MathAction]) -> np.ndarray:
    """
    Very lightweight lead‑lag path signature:
    cumulative sum of expected values followed by its first difference.
    """
    ev = np.array([a.expected_value for a in actions], dtype=np.float64)
    cum = np.cumsum(ev)
    diff = np.diff(cum, prepend=0.0)
    return np.concatenate([cum, diff])

def regret_weighted_utility(path_sig: np.ndarray, cockpit: np.ndarray) -> float:
    """
    Regret‑weighted utility U = (path_sig · cockpit) / (||path_sig||·||cockpit|| + ε)
    """
    eps = 1e-12
    dot = np.dot(path_sig, cockpit)
    norm = np.linalg.norm(path_sig) * np.linalg.norm(cockpit) + eps
    return dot / norm

# ----------------------------------------------------------------------
# Fractional calculus utilities (from Parent B)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for real z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1.0 - z))
    else:
        x = np.polyval(_LANCZOS_C, z)
        t = z + _LANCZOS_G + 0.5
        return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_fractional_derivative(history: np.ndarray, alpha: float, t: int) -> np.ndarray:
    """
    Discrete Caputo derivative of order alpha for a vector‑valued history.
    history must be a (t, N) array where each row is w(τ) for τ=0…t‑1.
    Returns an (N,) array approximating Dᶜα w(t).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    coeff = 1.0 / gamma_lanczos(1.0 - alpha)
    weights = np.array([1.0 / ((t - tau) ** alpha) for tau in range(t)], dtype=np.float64)
    diffs = history[1:] - history[:-1]  # w(τ+1) - w(τ)
    # Pad diffs with a zero row at the start to align dimensions
    diffs = np.vstack([np.zeros_like(diffs[0]), diffs])
    weighted = coeff * (weights[:, None] * diffs).sum(axis=0)
    return weighted

# ----------------------------------------------------------------------
# Hybrid core (new)
# ----------------------------------------------------------------------
@dataclass
class HybridState:
    """Encapsulates the internal state of the hybrid system."""
    weights: np.ndarray                # NLMS weight vector
    weight_history: List[np.ndarray]   # chronological list of past weights
    time_step: int = 0                 # current discrete time index

def nlms_fractional_update(state: HybridState,
                           input_vec: np.ndarray,
                           desired: float,
                           mu: float,
                           alpha: float,
                           utility: float,
                           eps: float = 1e-8) -> HybridState:
    """
    Perform one hybrid NLMS update where the step size μ is scaled by the
    regret‑weighted utility and the weight dynamics follow a Caputo fractional
    derivative of order α.
    """
    # Normalized LMS raw update
    norm_sq = np.dot(input_vec, input_vec) + eps
    error = desired - np.dot(state.weights, input_vec)
    raw_delta = (mu * utility) * error * input_vec / norm_sq

    # Append provisional weight (Euler step) to history for derivative calc
    provisional = state.weights + raw_delta
    new_history = state.weight_history + [provisional]

    # Compute fractional derivative approximation
    if state.time_step >= 1:
        frac_deriv = caputo_fractional_derivative(
            np.stack(new_history[-(state.time_step + 1):], axis=0),
            alpha,
            state.time_step + 1
        )
        # Blend Euler delta with fractional memory term
        new_weights = state.weights + raw_delta + frac_deriv
    else:
        new_weights = provisional  # first step, no memory yet

    # Update state
    return HybridState(
        weights=new_weights,
        weight_history=new_history,
        time_step=state.time_step + 1
    )

def hybrid_step(state: HybridState,
                actions: List[MathAction],
                morph: Morphology,
                mu: float = 0.5,
                alpha: float = 0.3) -> HybridState:
    """
    One complete hybrid iteration:
      1. Extract morphological features.
      2. Build a cockpit metric vector (simple linear transform of features).
      3. Compute path signature from recent actions.
      4. Derive regret‑weighted utility U.
      5. Feed U into the fractional NLMS weight update.
    Returns the updated state.
    """
    # 1. Morphology → feature vector
    morph_feat = compute_morphology_features(morph)          # (3,)

    # 2. Cockpit metrics (example: scaled features)
    cockpit = 2.0 * morph_feat + np.array([0.1, -0.2, 0.05])

    # 3. Path signature
    path_sig = compute_path_signature(actions)               # (2*len(actions),)

    # 4. Regret‑weighted utility
    utility = regret_weighted_utility(path_sig, cockpit)

    # 5. NLMS input vector = concatenation of features & path signature
    input_vec = np.concatenate([morph_feat, path_sig])
    desired = np.sum(path_sig) * 0.01  # synthetic desired signal

    # 6. Fractional NLMS update
    new_state = nlms_fractional_update(
        state,
        input_vec,
        desired,
        mu,
        alpha,
        utility
    )
    return new_state

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _generate_dummy_actions(n: int) -> List[MathAction]:
    """Create n random MathAction objects."""
    actions = []
    for i in range(n):
        ev = random.uniform(-1.0, 1.0)
        actions.append(
            MathAction(
                id=f"act{i}",
                tokens=("x", "y"),
                expected_value=ev,
                cost=random.random(),
                risk=random.random()
            )
        )
    return actions

def _initialize_state(dim_input: int) -> HybridState:
    """Initialize HybridState with zero weights and empty history."""
    init_weights = np.zeros(dim_input, dtype=np.float64)
    return HybridState(weights=init_weights, weight_history=[init_weights.copy()], time_step=0)

if __name__ == "__main__":
    # Create a dummy morphology
    morph = Morphology(length=2.5, width=1.8, height=0.9, mass=3.2)

    # Generate a short action sequence
    actions = _generate_dummy_actions(5)

    # Determine input dimension for NLMS (features + path signature)
    dummy_path = compute_path_signature(actions)
    input_dim = compute_morphology_features(morph).size + dummy_path.size

    # Initialise hybrid state
    state = _initialize_state(input_dim)

    # Run a few hybrid steps
    for step in range(8):
        state = hybrid_step(state, actions, morph, mu=0.4, alpha=0.35)
        print(f"Step {step+1:02d} | Weights norm: {np.linalg.norm(state.weights):.6f}")

    print("Hybrid fusion execution completed without errors.")