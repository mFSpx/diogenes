# DARWIN HAMMER — match 4575, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m170_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m1311_s1.py (gen6)
# born: 2026-05-29T23:56:37Z

"""Hybrid Algorithm integrating:
- Parent A: `hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m170_s0.py`
  (ternary vector generation + decision‑hygiene scoring used for logistic margins)
- Parent B: `hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m1311_s1.py`
  (Liquid‑Time‑Constant recurrent dynamics with a ternary‑linear weight matrix)

Mathematical bridge:
The ternary vector produced by Parent A is used as the initial hidden state of the
Liquid‑Time‑Constant (LTC) cell from Parent B.  The decision‑hygiene score from
Parent A modulates the LTC non‑linearity scaling `α` **and** scales the logistic
margin that appears in the XGBoost‑style gradient/Hessian computation.
Thus a single scalar (`hygiene_score`) couples the high‑level decision quality
to both the recurrent dynamics and the loss‑gradient computation, yielding a
coherent hybrid system.""" 

import json
import hashlib
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants shared by both parents
# ----------------------------------------------------------------------
TERNARY_DIMS = 12               # dimensionality of the ternary vector / state
TAU = 1.0                       # base time constant for LTC
BASE_ALPHA = 0.5                # base non‑linearity scaling for LTC
DT = 0.01                       # integration step for LTC dynamics
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
# ----------------------------------------------------------------------
# Utility helpers (borrowed from Parent A)
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def payload_hash(raw_command: str, normalized_intent: str, context: dict) -> str:
    """Deterministic SHA‑256 hash of the command envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(raw_command: str, normalized_intent: str, context: dict) -> np.ndarray:
    """
    Produce a deterministic ternary vector (values in {‑1, 0, 1}) from the payload.
    The algorithm mirrors Parent A's approach: bits of the SHA‑256 hash are
    interpreted in base‑4 and mapped to ternary symbols.
    """
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    hash_int = int(hashlib.sha256(encoded).hexdigest(), 16)

    vec = np.empty(TERNARY_DIMS, dtype=int)
    for i in range(TERNARY_DIMS):
        # extract two bits → value in {0,1,2,3}
        v = (hash_int >> (2 * i)) & 0b11
        if v == 0:
            vec[i] = -1
        elif v == 1:
            vec[i] = 0
        else:                     # v == 2 or 3 -> map to +1
            vec[i] = 1
    return vec


def decision_hygiene_score(raw_command: str, normalized_intent: str, context: dict) -> float:
    """
    A deterministic “hygiene” score in [0, 1] derived from the same payload.
    Parent A used a more elaborate scoring scheme; here we use a simple
    hash‑based fraction for reproducibility.
    """
    h = payload_hash(raw_command, normalized_intent, context)
    # Take first 8 hex digits → 32‑bit integer → normalize
    score_int = int(h[:8], 16)
    return (score_int % 1000) / 1000.0  # yields a float in [0, 0.999]


# ----------------------------------------------------------------------
# Components from Parent B (LTC dynamics + ternary‑linear matrix)
# ----------------------------------------------------------------------
def ternary_linear_weight_matrix(dim: int, seed: int = 0) -> np.ndarray:
    """
    Generate a square weight matrix with entries in {‑1, 0, 1}.
    The matrix is deterministic given ``dim`` and ``seed``.
    """
    rng = np.random.default_rng(seed)
    # Sample from {-1,0,1} uniformly
    return rng.choice([-1, 0, 1], size=(dim, dim))


def ltc_step(
    state: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    dt: float = DT,
    tau: float = TAU,
    alpha: float = BASE_ALPHA,
) -> np.ndarray:
    """
    One explicit Euler step of a Liquid‑Time‑Constant (LTC) recurrent cell:

        dx/dt = (‑x + f(W·x + b)) / τ
        f(z) = tanh(α·z)

    The update rule becomes:

        x_{t+1} = x_t + dt * (‑x_t + tanh(α·(W·x_t + b))) / τ
    """
    linear = weight @ state + bias
    nonlinear = np.tanh(alpha * linear)
    derivative = (-state + nonlinear) / tau
    return state + dt * derivative


def run_ltc_dynamics(
    init_state: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    steps: int = 10,
    dt: float = DT,
    tau: float = TAU,
    alpha: float = BASE_ALPHA,
) -> np.ndarray:
    """Iterate ``steps`` LTC updates and return the final state."""
    state = init_state.astype(float)
    for _ in range(steps):
        state = ltc_step(state, weight, bias, dt, tau, alpha)
    return state


# ----------------------------------------------------------------------
# Hybrid logistic objective (XGBoost‑style) – Parent A meets Parent B
# ----------------------------------------------------------------------
def logistic_margin(
    ternary_vec: np.ndarray,
    weight: np.ndarray,
    hygiene_score: float,
) -> float:
    """
    Compute a scalar margin that feeds the binary logistic loss.
    The margin is the dot product between the ternary vector and the mean
    column of the weight matrix, scaled by the hygiene score.
    """
    mean_w = weight.mean(axis=1)               # shape (dim,)
    raw_margin = float(np.dot(ternary_vec, mean_w))
    return hygiene_score * raw_margin


def logistic_gradient_hessian(
    pred: float,
    label: int,
    margin: float,
) -> Tuple[float, float]:
    """
    Binary logistic gradient and hessian using a margin.
    The prediction ``pred`` is the raw score (before sigmoid).
    """
    # Apply margin as an additive shift
    z = pred + margin
    prob = 1.0 / (1.0 + math.exp(-z))
    grad = prob - label
    hess = prob * (1.0 - prob)
    return grad, hess


# ----------------------------------------------------------------------
# Public API – three demonstration functions
# ----------------------------------------------------------------------
def demo_vector_and_score(raw_command: str, normalized_intent: str, context: dict) -> Tuple[np.ndarray, float]:
    """
    Generate the ternary vector and the decision‑hygiene score.
    Returns (vector, score).
    """
    vec = ternary_vector(raw_command, normalized_intent, context)
    score = decision_hygiene_score(raw_command, normalized_intent, context)
    return vec, score


def demo_ltc_dynamics(raw_command: str, normalized_intent: str, context: dict) -> np.ndarray:
    """
    Run the LTC recurrent dynamics using the ternary vector as the initial state.
    The hygiene score modulates the non‑linearity scaling α.
    """
    vec, hygiene = demo_vector_and_score(raw_command, normalized_intent, context)
    weight = ternary_linear_weight_matrix(TERNARY_DIMS, seed=42)
    bias = np.zeros(TERNARY_DIMS)
    alpha_mod = BASE_ALPHA * (1.0 + hygiene)  # hygiene enlarges the effective α
    final_state = run_ltc_dynamics(
        init_state=vec,
        weight=weight,
        bias=bias,
        steps=20,
        alpha=alpha_mod,
    )
    return final_state


def demo_hybrid_objective(
    raw_command: str,
    normalized_intent: str,
    context: dict,
    label: int,
) -> Tuple[float, float]:
    """
    End‑to‑end demonstration:
    1. Produce ternary vector and hygiene score.
    2. Build a ternary‑linear weight matrix.
    3. Compute a logistic margin that blends both sources.
    4. Return gradient and hessian of the logistic loss for the given label.
    """
    vec, hygiene = demo_vector_and_score(raw_command, normalized_intent, context)
    weight = ternary_linear_weight_matrix(TERNARY_DIMS, seed=123)
    margin = logistic_margin(vec, weight, hygiene)
    # Use the raw dot‑product as the base prediction (no sigmoid yet)
    base_pred = float(np.dot(vec, weight.mean(axis=1)))
    grad, hess = logistic_gradient_hessian(base_pred, label, margin)
    return grad, hess


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Mock payload
    raw_cmd = "run analysis"
    intent = "analysis"
    ctx = {"user": "alice", "session": "42"}

    # Demo 1: vector & hygiene
    vec, score = demo_vector_and_score(raw_cmd, intent, ctx)
    print("Ternary vector:", vec)
    print("Hygiene score:", round(score, 4))

    # Demo 2: LTC dynamics
    final_state = demo_ltc_dynamics(raw_cmd, intent, ctx)
    print("Final LTC state (first 5 components):", final_state[:5])

    # Demo 3: Hybrid logistic objective
    grad, hess = demo_hybrid_objective(raw_cmd, intent, ctx, label=1)
    print("Logistic gradient:", round(grad, 6))
    print("Logistic hessian :", round(hess, 6))

    print("All demos executed successfully at", utc_now())