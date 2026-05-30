# DARWIN HAMMER — match 5089, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1035_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m2639_s0.py (gen4)
# born: 2026-05-29T23:59:53Z

"""Hybrid Algorithm integrating:
- Parent A: regex‑based evidence/planning feature extraction (hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s2.py)
- Parent B: TTT‑Linear adaptive compression with Caputo fractional memory kernel
  (hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m2639_s0.py)

Mathematical Bridge
-------------------
Parent A yields a 2‑element feature column vector **v** = [evidence, planning]ᵀ.
A scalar confidence ρ ∈ (0,1) is derived from **v** (ρ = ‖v‖ / (‖v‖ + 1)).
Parent B updates a weight matrix **W** by gradient descent on the TTT loss.
The hybrid blends the two by
    ΔW = ρ · η · ( Σ_{k=0}^{n-1} κ_k·g_k )
where g_k are past gradients, η is a learning rate, and κ_k are
Caputo fractional kernel weights κ_k = ( (n‑k)^{‑α} ) / Γ(1‑α) with 0<α<1.
Thus the feature‑derived confidence scales a fractional‑memory‑aware
gradient step, fusing textual evidence with a morphology‑like memory
process.
"""

import sys
import random
import math
import pathlib
import numpy as np
import re
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Parent A – regex feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def _extract_counts(text: str) -> dict:
    """Return raw counts of evidence‑related and planning‑related tokens."""
    evidence = len(EVIDENCE_RE.findall(text))
    planning = len(PLANNING_RE.findall(text))
    return {"evidence": evidence, "planning": planning}


def feature_vector(text: str) -> np.ndarray:
    """2‑element column vector [evidence; planning] as float64."""
    cnts = _extract_counts(text)
    return np.array([cnts["evidence"], cnts["planning"]], dtype=float).reshape(2, 1)


def confidence_from_vector(v: np.ndarray) -> float:
    """
    Map a 2‑element column vector to a scalar confidence ρ ∈ (0,1).

    ρ = ‖v‖ / (‖v‖ + 1)  ensures smooth saturation.
    """
    norm = np.linalg.norm(v)
    return norm / (norm + 1.0)


# ----------------------------------------------------------------------
# Parent B – TTT‑Linear core with gradient descent
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix W ~ N(0,scale²)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Quadratic loss L = ||W·x – target||² (target defaults to x)."""
    if target is None:
        target = x
    diff = W @ x - target
    return float(np.sum(diff ** 2))


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient ∂L/∂W = 2·(W·x – target)·xᵀ."""
    if target is None:
        target = x
    diff = W @ x - target
    return 2.0 * diff @ x.T


# ----------------------------------------------------------------------
# Fractional (Caputo) memory utilities
# ----------------------------------------------------------------------
def caputo_kernel(alpha: float, length: int) -> np.ndarray:
    """
    Return Caputo power‑law kernel κ_k = ( (k+1)^{-α} ) / Γ(1‑α)
    for k = 0 … length‑1.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    coeff = 1.0 / math.gamma(1.0 - alpha)
    ks = np.arange(1, length + 1, dtype=float)
    return coeff * ks ** (-alpha)


def fractional_weighted_sum(grad_history: list[np.ndarray], alpha: float) -> np.ndarray:
    """
    Compute Σ κ_k·g_k where g_k are stored gradients ordered from newest to oldest.
    """
    if not grad_history:
        raise ValueError("gradient history is empty")
    kernel = caputo_kernel(alpha, len(grad_history))
    weighted = sum(k * g for k, g in zip(kernel, grad_history))
    return weighted


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_step(
    W: np.ndarray,
    x: np.ndarray,
    eta: float,
    alpha: float,
    grad_history: list[np.ndarray],
    confidence: float,
) -> np.ndarray:
    """
    Perform one hybrid update:
        g   = gradient of current loss
        push g onto history (most recent first)
        G   = fractional weighted sum of history
        ΔW  = - confidence * eta * G
        W'  = W + ΔW
    Returns the updated weight matrix.
    """
    # Current gradient
    g = ttt_grad(W, x)
    # Update history (keep newest at index 0)
    grad_history.insert(0, g)
    # Optional truncation to keep memory manageable
    max_len = 50
    if len(grad_history) > max_len:
        grad_history.pop()
    # Fractional aggregation
    G = fractional_weighted_sum(grad_history, alpha)
    # Scaled update
    delta = -confidence * eta * G
    return W + delta


def hybrid_process_text(
    text: str,
    W: np.ndarray,
    eta: float,
    alpha: float,
    grad_history: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    End‑to‑end hybrid processing for a single document:
        1. Extract feature vector v.
        2. Compute confidence ρ from v.
        3. Treat v as input x to the TTT system.
        4. Update W with hybrid_step.
    Returns the new weight matrix and the confidence used.
    """
    v = feature_vector(text)               # shape (2,1)
    rho = confidence_from_vector(v)        # scalar ∈ (0,1)
    W_new = hybrid_step(W, v, eta, alpha, grad_history, rho)
    return W_new, rho


def evaluate_hybrid(
    W: np.ndarray,
    texts: list[str],
    eta: float,
    alpha: float,
) -> dict:
    """
    Run the hybrid system over a list of texts, accumulating loss and
    returning a summary dictionary.
    """
    grad_history: list[np.ndarray] = []
    total_loss = 0.0
    for txt in texts:
        v = feature_vector(txt)
        rho = confidence_from_vector(v)
        loss = ttt_loss(W, v)
        total_loss += loss
        W = hybrid_step(W, v, eta, alpha, grad_history, rho)
    return {"final_weights": W, "average_loss": total_loss / len(texts)}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts with varying evidence/planning tokens
    sample_texts = [
        "The audit confirmed the evidence and the plan was executed.",
        "No source or checklist was provided.",
        "Verified record, but missing roadmap and schedule.",
        "Proof of concept and test steps are documented.",
    ]

    # Initialise a 2×2 weight matrix (matching the 2‑dimensional feature space)
    W0 = init_ttt(d_in=2, scale=0.05, seed=42)

    # Hyper‑parameters
    learning_rate = 0.01
    fractional_alpha = 0.6  # memory exponent

    # Run evaluation
    result = evaluate_hybrid(W0, sample_texts, learning_rate, fractional_alpha)

    # Simple diagnostics
    print("Final weight matrix:\n", result["final_weights"])
    print("Average loss:", result["average_loss"])
    # Verify that the code runs without raising exceptions
    sys.exit(0)