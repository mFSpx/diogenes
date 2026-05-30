# DARWIN HAMMER — match 3828, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1263_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m2604_s0.py (gen4)
# born: 2026-05-29T23:51:49Z

"""Hybrid LSM‑Bandit‑Allocation‑Audit Fusion

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1263_s1.py (NLMS‑adapted bandit dynamics)
- hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m2604_s0.py (allocation + sheaf‑cohomology audit)

Mathematical bridge
------------------
Let **g** = (g₀,…,g_{n‑1}) be the ordered list of groups.
Parent B produces a 0‑cochain **s**∈ℝⁿ (allocation scores) and a penalty
vector **p**∈ℝⁿ (audit penalties).  
Parent A maintains a weight vector **w**∈ℝⁿ that drives bandit propensities.
We fuse the two by letting the Hadamard product **u = (λ·s) ⊙ p** be the
input signal **x** for the Normalised Least‑Mean‑Squares (NLMS) update that
adapts **w**:

  **w⁺ = w + μ · (e) · x / (‖x‖²+ε)** , e = τ – **w·x** ,

where τ is a scalar target (chosen as 1.0) and μ∈(0,1) a learning rate.
The updated weights are turned into bandit propensities via a soft‑max.
A coboundary matrix **B** (δ:C⁰→C¹) maps the weighted section **v = s ⊙ p**
to edge‑wise differences; the L²‑norm ‖B v‖₂ quantifies topological
inconsistency and is returned as a residual.

The module implements the full hybrid loop in `hybrid_step`, exposing the
core operations as separate functions. """

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import datetime as dt

# ----------------------------------------------------------------------
# Shared constants (from Parent B)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

# ----------------------------------------------------------------------
# Parent A – NLMS / bandit utilities
# ----------------------------------------------------------------------


def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float = 1.0,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised Least‑Mean‑Squares weight adaptation.

    Returns the updated weight vector and the instantaneous error.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + mu * error * x / power
    return new_weights, error


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable soft‑max."""
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    return exp_z / exp_z.sum()


# ----------------------------------------------------------------------
# Parent B – allocation / audit / sheaf utilities
# ----------------------------------------------------------------------


def weekday_weight_vector() -> np.ndarray:
    """
    Produce a 4‑element vector of weekday‑dependent scalars.
    Monday → 1.0, Tuesday → 0.9, …, Sunday → 0.4 (example scheme).
    """
    day = dt.datetime.utcnow().weekday()  # 0 = Monday
    base = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    return np.full(len(GROUPS), base[day % 7])


def allocate_hybrid(groups: Tuple[str, ...], weekday_weights: np.ndarray) -> np.ndarray:
    """
    Deterministic allocation per group using weekday weights.
    The allocation score for group i is proportional to the weekday weight
    multiplied by a fixed group bias (here set to 1.0 for simplicity).
    """
    bias = np.ones(len(groups))
    allocation = bias * weekday_weights
    # Normalise to sum to 1 for interpretability
    return allocation / allocation.sum()


def audit_penalty_vector(groups: Tuple[str, ...], seed: int = 42) -> np.ndarray:
    """
    Mock audit routine: generate a reproducible penalty between 0 and 1
    for each group. In a real system this would aggregate audit findings.
    """
    rng = np.random.default_rng(seed)
    return rng.random(len(groups))


def decreasing_prune_factor(step: int, init: float = 1.0, decay: float = 0.99) -> float:
    """Geometric decay λ(step) = init * decay**step."""
    return float(init * (decay ** step))


def coboundary_matrix(n: int) -> np.ndarray:
    """
    Construct a simple coboundary operator δ : C⁰ → C¹ for a chain of n vertices.
    The matrix has (n‑1) rows, each encoding the difference between consecutive
    vertices: row i = e_i = (0,…,1,‑1,…,0).
    """
    if n < 2:
        return np.zeros((0, n))
    B = np.zeros((n - 1, n))
    for i in range(n - 1):
        B[i, i] = 1.0
        B[i, i + 1] = -1.0
    return B


# ----------------------------------------------------------------------
# Hybrid system
# ----------------------------------------------------------------------


@dataclass
class HybridState:
    """Container for the evolving weight vector and step counter."""
    weights: np.ndarray
    step: int = 0


def hybrid_step(state: HybridState, groups: Tuple[str, ...]) -> Tuple[HybridState, Dict]:
    """
    Perform one hybrid iteration:

    1. Compute allocation scores `s` from weekday weights.
    2. Compute audit penalties `p`.
    3. Apply decreasing‑rate pruning λ(step) to `s`.
    4. Form the input signal `x = (λ·s) ⊙ p`.
    5. Update bandit weights via NLMS using target τ=1.0.
    6. Derive propensities via soft‑max of the updated weights.
    7. Compute sheaf residual ‖δ (s ⊙ p)‖₂.

    Returns the new state and a diagnostics dictionary.
    """
    # 1. Allocation scores
    weekday_w = weekday_weight_vector()
    s = allocate_hybrid(groups, weekday_w)  # shape (n,)

    # 2. Audit penalties
    p = audit_penalty_vector(groups, seed=state.step)  # deterministic per step

    # 3. Pruning factor
    lam = decreasing_prune_factor(state.step)

    # 4. Input signal for NLMS
    x = (lam * s) * p  # Hadamard product

    # 5. NLMS weight update
    new_weights, error = nlms_update(state.weights, x, target=1.0, mu=0.5)

    # 6. Propensities (soft‑max)
    propensities = softmax(new_weights)

    # 7. Sheaf residual
    B = coboundary_matrix(len(groups))
    weighted_section = s * p
    residual = np.linalg.norm(B @ weighted_section) if B.size else 0.0

    diagnostics = {
        "step": state.step,
        "lambda": lam,
        "allocation": s,
        "penalties": p,
        "input_signal": x,
        "nlms_error": error,
        "propensities": propensities,
        "sheaf_residual": residual,
    }

    new_state = HybridState(weights=new_weights, step=state.step + 1)
    return new_state, diagnostics


# ----------------------------------------------------------------------
# Simple graph construction (placeholder from Parent A)
# ----------------------------------------------------------------------


def construct_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a naive directed graph where each node i connects to i+1 with
    edge weight equal to the absolute difference of the corresponding
    bandit weights. The graph is represented as an adjacency list.
    """
    n = len(weights)
    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n - 1):
        w_diff = float(abs(weights[i + 1] - weights[i]))
        graph.setdefault(i, []).append((i + 1, w_diff))
    return graph


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Initialise with zero weights
    init_weights = np.zeros(len(GROUPS))
    state = HybridState(weights=init_weights)

    # Run a few hybrid steps and print diagnostics
    for _ in range(5):
        state, info = hybrid_step(state, GROUPS)
        print(f"Step {info['step']}: λ={info['lambda']:.4f}, "
              f"error={info['nlms_error']:.4f}, residual={info['sheaf_residual']:.4f}")
        print(f"  Propensities: {np.round(info['propensities'], 3)}")
        # Optional: inspect the graph built from the latest weights
        g = construct_graph(state.weights)
        print(f"  Graph edges: {g}\n")