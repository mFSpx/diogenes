# DARWIN HAMMER — match 5630, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1263_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_sparse_wta_hy_m1093_s1.py (gen3)
# born: 2026-05-30T00:03:39Z

"""Hybrid Perceptron‑KAN‑Store‑SparseWTA Fusion
================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1263_s0.py``  
  Provides an online perceptron (linear regression) together with a
  “store” dynamical system whose inflow/outflow gains are denoted by
  ``alpha`` and ``beta``.  The store level ``ℓ`` evolves as  

      ℓₜ₊₁ = ℓₜ + α·inflowₜ – β·outflowₜ

  and the store level is later turned into a gating factor
  ``g = sigmoid(ℓ)``.

* **Parent B** – ``hybrid_hybrid_hybrid_hard_t_hybrid_sparse_wta_hy_m1093_s1.py``  
  Implements a single‑layer Kolmogorov‑Arnold‑Network (KAN) that maps a
  stylometric feature vector ``s`` to ``y`` using univariate B‑splines,
  followed by a deterministic hash‑expansion, a top‑k winner‑take‑all
  (WTA) sparsification and a privacy‑risk scalar derived from the
  normalized Hamming distance between the resulting binary mask and a
  reference mask.  The risk scalar scales Laplace noise added to a final
  decision.

Mathematical Bridge
-------------------
The fusion is built on two reciprocal couplings:

1. **Perceptron → KAN inflow** – The perceptron prediction
   ``p = wᵀ·x`` is used as the *inflow* term of the store dynamics.
2. **Store level → KAN gating** – The store level ``ℓ`` is passed through
   a sigmoid to obtain a gating factor ``g ∈ (0,1)``.  The KAN output
   ``y`` is multiplied by ``g`` before the sparse‑WTA stage, thus the
   store controls the amplitude of the representation.
3. **Sparse‑WTA → Store outflow** – The Hamming‑based privacy risk ``r`` is
   fed back as the *outflow* term of the store, closing the loop.

The resulting closed‑loop system simultaneously learns a linear predictor,
produces a spline‑based nonlinear embedding, enforces sparsity for privacy,
and regulates the whole process through a simple store dynamical system.

The module below implements this hybrid pipeline and provides three
illustrative functions.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    """Simple store (integrator) with inflow/outflow gains."""
    level: float = 0.0          # ℓₜ
    alpha: float = 1.0          # inflow gain α
    beta: float = 1.0           # outflow gain β
    dt: float = 1.0             # time step (unused but kept for compatibility)

    def gating(self) -> float:
        """Sigmoid gating factor derived from the current store level."""
        return 1.0 / (1.0 + math.exp(-self.level))


# ----------------------------------------------------------------------
# Parent A – Perceptron utilities
# ----------------------------------------------------------------------


def perceptron_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = wᵀ·x."""
    return float(np.dot(weights, x))


def perceptron_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    LMS‑style weight update.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Instantaneous prediction error (target − prediction).
    """
    y_hat = perceptron_predict(weights, x)
    error = target - y_hat
    power = float(np.dot(x, x) + eps)
    new_weights = weights + mu * error * x / power
    return new_weights, error


# ----------------------------------------------------------------------
# Parent B – KAN + Sparse WTA utilities
# ----------------------------------------------------------------------


def b_spline_basis(s: float, tau: float, degree: int = 3) -> float:
    """
    Very simplified univariate B‑spline basis of given degree.
    For the purpose of this hybrid demo we use a triangular kernel:

        B(s; τ) = max(0, 1 - |s-τ|/degree)

    This satisfies the required locality and smoothness qualitatively.
    """
    return max(0.0, 1.0 - abs(s - tau) / float(degree))


def kan_transform(
    s: np.ndarray,
    weight_matrix: np.ndarray,
    knots: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """
    Single‑layer KAN mapping.

    Parameters
    ----------
    s : np.ndarray, shape (d,)
        Input feature vector.
    weight_matrix : np.ndarray, shape (d, m)
        Linear coefficients w_{ij}.
    knots : np.ndarray, shape (d,)
        Knot positions τ_i for each input dimension.
    degree : int
        Degree of the (simplified) B‑spline basis.

    Returns
    -------
    y : np.ndarray, shape (m,)
        KAN output.
    """
    d, m = weight_matrix.shape
    assert s.shape[0] == d
    # Compute basis vector B_i = B(s_i; τ_i)
    B = np.array([b_spline_basis(s[i], knots[i], degree) for i in range(d)])
    # Linear combination per output dimension
    y = B @ weight_matrix  # shape (m,)
    return y


def expand_hash(y: np.ndarray, M: int) -> np.ndarray:
    """
    Deterministic hash‑based expansion of a low‑dimensional vector
    to a higher dimensional space.  The implementation repeats the
    vector and truncates to length M.
    """
    if M <= 0:
        raise ValueError("Target dimension M must be positive")
    repeats = int(np.ceil(M / y.size))
    expanded = np.tile(y, repeats)[:M]
    return expanded


def top_k_mask(vec: np.ndarray, k: int) -> np.ndarray:
    """
    Winner‑take‑all sparsification: returns a binary mask of length
    ``len(vec)`` with ones at the indices of the top‑k largest values.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k > vec.size:
        k = vec.size
    mask = np.zeros(vec.shape, dtype=np.int8)
    top_indices = np.argpartition(-vec, k - 1)[:k]
    mask[top_indices] = 1
    return mask


def normalized_hamming(mask: np.ndarray, reference: np.ndarray) -> float:
    """
    Normalized Hamming distance between two binary masks.
    Returns a value in [0, 1].
    """
    if mask.shape != reference.shape:
        raise ValueError("Mask and reference must have the same shape")
    diff = np.sum(mask != reference)
    return diff / float(mask.size)


def laplace_noise(scale: float = 1.0) -> float:
    """Sample zero‑mean Laplace noise."""
    return float(np.random.laplace(loc=0.0, scale=scale))


# ----------------------------------------------------------------------
# Fusion primitives
# ----------------------------------------------------------------------


def store_update(
    state: StoreState,
    inflow: float,
    outflow: float,
) -> StoreState:
    """
    Integrate the store dynamics for one time step.

    ℓₜ₊₁ = ℓₜ + α·inflow – β·outflow
    """
    new_level = state.level + state.alpha * inflow - state.beta * outflow
    return StoreState(level=new_level, alpha=state.alpha, beta=state.beta, dt=state.dt)


def hybrid_step(
    state: StoreState,
    percep_weights: np.ndarray,
    x: np.ndarray,
    target: float,
    kan_weights: np.ndarray,
    knots: np.ndarray,
    ref_mask: np.ndarray,
    M: int,
    k: int,
    degree: int = 3,
) -> Tuple[StoreState, np.ndarray, float]:
    """
    Execute one hybrid iteration:

    1. Perceptron prediction → inflow.
    2. Update store level.
    3. Compute gating g = sigmoid(ℓ).
    4. KAN transform of the same input ``x`` (treated as stylometric vector).
    5. Gate KAN output with ``g``.
    6. Expand, apply top‑k WTA, compute privacy risk ``r``.
    7. Use ``r`` as outflow for the next store update.
    8. Return updated state, updated perceptron weights, and a final
       decision value (gated KAN output summed + Laplace noise scaled by r).

    Returns
    -------
    new_state : StoreState
        Updated store after accounting for inflow and outflow.
    new_weights : np.ndarray
        Updated perceptron weight vector.
    decision : float
        Privacy‑aware decision scalar.
    """
    # ---- 1. Perceptron prediction (inflow) ----
    inflow = perceptron_predict(percep_weights, x)

    # ---- 2. Preliminary store update (using previous outflow=0) ----
    # We will apply the true outflow after computing risk.
    prelim_state = store_update(state, inflow, outflow=0.0)

    # ---- 3. Gating factor from store level ----
    g = prelim_state.gating()

    # ---- 4. KAN transformation ----
    y = kan_transform(s=x, weight_matrix=kan_weights, knots=knots, degree=degree)

    # ---- 5. Apply gating ----
    y_gated = g * y

    # ---- 6. Sparse WTA encoding ----
    expanded = expand_hash(y_gated, M)
    mask = top_k_mask(expanded, k)

    # ---- 7. Privacy risk (outflow) ----
    risk = normalized_hamming(mask, ref_mask)          # r ∈ [0,1]
    outflow = risk                                      # direct mapping

    # ---- 8. Final store update with true outflow ----
    new_state = store_update(state, inflow, outflow)

    # ---- 9. Perceptron weight update (using same target) ----
    new_weights, _ = perceptron_update(percep_weights, x, target)

    # ---- 10. Decision value (privacy‑aware) ----
    decision = float(np.sum(y_gated)) + laplace_noise(scale=1.0 * risk)

    return new_state, new_weights, decision


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------


def demo_perceptron_kan():
    """Show that a perceptron weight vector can be re‑used as a KAN matrix."""
    d, m = 5, 4
    np.random.seed(0)
    w = np.random.randn(d)                     # perceptron weights
    kan_w = np.outer(w, np.ones(m)) / np.sqrt(m)  # simple broadcasting to shape (d,m)
    knots = np.linspace(-1, 1, d)

    x = np.random.randn(d)
    target = 0.0

    # Perceptron step
    w_new, err = perceptron_update(w, x, target)

    # KAN step (no gating)
    y = kan_transform(x, kan_w, knots)

    print("Perceptron error:", err)
    print("KAN output:", y)


def demo_store_sparse():
    """Run the store dynamics together with sparse WTA without the perceptron loop."""
    state = StoreState(level=0.2, alpha=0.8, beta=0.5)
    inflow = 1.2
    outflow = 0.3
    new_state = store_update(state, inflow, outflow)
    print("Old level:", state.level, "New level:", new_state.level)

    vec = np.array([0.1, 0.7, -0.2, 0.4, 0.9])
    M, k = 10, 3
    mask = top_k_mask(expand_hash(vec, M), k)
    ref = np.zeros_like(mask)
    risk = normalized_hamming(mask, ref)
    print("Mask:", mask)
    print("Privacy risk (Hamming):", risk)


def demo_full_hybrid():
    """Execute a few hybrid iterations and print the evolving state."""
    np.random.seed(42)

    # Dimensions
    d = 6          # input / stylometric dimension
    m = 8          # KAN output dimension
    M = 32         # expanded dimension
    k = 5          # WTA sparsity

    # Initialise components
    percep_w = np.random.randn(d)
    kan_w = np.random.randn(d, m) * 0.1
    knots = np.linspace(-1.0, 1.0, d)

    ref_mask = np.zeros(M, dtype=np.int8)          # reference all‑zero mask
    state = StoreState(level=0.0, alpha=0.9, beta=0.7)

    # Simulated data stream
    for step in range(5):
        x = np.random.randn(d)
        target = float(np.sum(x)) * 0.3  # synthetic target

        state, percep_w, decision = hybrid_step(
            state=state,
            percep_weights=percep_w,
            x=x,
            target=target,
            kan_weights=kan_w,
            knots=knots,
            ref_mask=ref_mask,
            M=M,
            k=k,
            degree=3,
        )
        print(
            f"Step {step+1:2d} | Store level={state.level: .4f} | "
            f"Gating={state.gating(): .4f} | Decision={decision: .4f}"
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Perceptron ↔ KAN ===")
    demo_perceptron_kan()
    print("\n=== Demo: Store + Sparse WTA ===")
    demo_store_sparse()
    print("\n=== Demo: Full Hybrid System ===")
    demo_full_hybrid()