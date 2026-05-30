# DARWIN HAMMER — match 158, survivor 1
# gen: 3
# parent_a: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# parent_b: hybrid_hybrid_model_vram_sc_fold_change_detectio_m32_s0.py (gen2)
# born: 2026-05-29T23:25:54Z

"""Hybrid Liquid‑Time‑Constant, MinHash, VRAM‑Scheduler & Fold‑Change Detection.

Parents
-------
* **Parent A** – `hybrid_liquid_time_constant_minhash_m10_s2.py`  
  Provides a Liquid‑Time‑Constant (LTC) recurrent dynamics whose effective
  time‑constant τ_eff is modulated by a MinHash similarity signal *s_t* between
  consecutive token sets.

* **Parent B** – `hybrid_hybrid_model_vram_sc_fold_change_detectio_m32_s0.py`  
  Supplies a VRAM‑scheduler‑style weight matrix *W* that is updated by gradient
  descent and a Fold‑Change Detection (FCD) mechanism that computes a scalar
  *c_t* from the relative change of an external input *I*.

Mathematical Bridge
-------------------
Both parents describe recurrent systems whose updates are driven by a scalar
modulator:

* In A the modulator is `f(x_t,I_t,θ) + α·s_t`.
* In B the modulator is the fold‑change term `c_t = (I_t‑I_{t‑1}) / (|I_{t‑1}|+ε)`.

We fuse them by **adding** the two modulators inside the LTC effective time
constant and by letting the same fold‑change scalar also scale the learning‑rate
used for the VRAM‑scheduler weight update.  The resulting hybrid step is


τ_eff(t) = τ / (1 + τ·( f_t + α·s_t + β·c_t ))
dx/dt   = -(1/τ + f_t + α·s_t + β·c_t)·x_t + (f_t + α·s_t + β·c_t)·(W_t @ x_t)
W_{t+1} = W_t - η·(1+γ·c_t)·∂L/∂W_t          (simple outer‑product surrogate)


where `f_t = sigmoid(θ·[x_t;I_t])`.  This single unified system reacts to
textual similarity, to abrupt input changes, and continuously adapts its weight
matrix.

The module below implements the fused dynamics and provides three public
functions demonstrating the hybrid operation.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# MinHash utilities (Parent A)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
DEFAULT_NUM_PERM = 64  # length of MinHash signature


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def shingles(text: str, k: int = 5) -> List[str]:
    """Return a list of k‑shingles (contiguous token substrings)."""
    tokens = text.split()
    return [" ".join(tokens[i : i + k]) for i in range(max(len(tokens) - k + 1, 1))]


def minhash_signature(tokens: Iterable[str], num_perm: int = DEFAULT_NUM_PERM) -> np.ndarray:
    """Compute a MinHash signature of length ``num_perm`` for the given tokens."""
    sig = np.full(num_perm, MAX64, dtype=np.uint64)
    for token in tokens:
        for i in range(num_perm):
            h = _hash(i, token)
            if h < sig[i]:
                sig[i] = h
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signatures must have the same length")
    return np.mean(sig1 == sig2)


# ----------------------------------------------------------------------
# Fold‑Change Detection utility (Parent B)
# ----------------------------------------------------------------------
def fold_change(current: float, previous: float, eps: float = 1e-12) -> float:
    """Return the relative change (fold‑change) between two scalar inputs."""
    return (current - previous) / (abs(previous) + eps)


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant gating (Parent A)
# ----------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ltc_gate(x: np.ndarray, I: float, theta: np.ndarray) -> float:
    """
    Gating function f(x_t, I_t, θ) = sigmoid(θᵀ·[x_t; I_t]).
    θ is a vector of length ``x.size + 1`` (last entry multiplies the scalar I).
    """
    aug = np.concatenate([x, np.array([I])])
    return float(sigmoid(theta @ aug))


# ----------------------------------------------------------------------
# Hybrid step combining LTC, MinHash, VRAM‑scheduler weight update,
# and Fold‑Change Detection.
# ----------------------------------------------------------------------
def hybrid_step(
    x_prev: np.ndarray,
    I_curr: float,
    I_prev: float,
    W: np.ndarray,
    theta: np.ndarray,
    tau: float,
    alpha: float,
    beta: float,
    eta: float,
    gamma: float,
    dt: float,
    sig_prev: np.ndarray,
    tokens_curr: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one hybrid update.

    Returns
    -------
    x_new : np.ndarray
        Updated hidden state.
    W_new : np.ndarray
        Updated weight matrix.
    sig_new : np.ndarray
        MinHash signature of the current token set (to be cached for next step).
    """
    # ---- MinHash similarity -------------------------------------------------
    sig_curr = minhash_signature(tokens_curr, num_perm=sig_prev.shape[0])
    s_t = minhash_similarity(sig_prev, sig_curr)  # ∈ [0,1]

    # ---- Fold‑change ---------------------------------------------------------
    c_t = fold_change(I_curr, I_prev)  # can be negative or positive

    # ---- Gating -------------------------------------------------------------
    f_t = ltc_gate(x_prev, I_curr, theta)  # scalar in (0,1)

    # ---- Effective time constant --------------------------------------------
    mod = f_t + alpha * s_t + beta * c_t
    tau_eff = tau / (1.0 + tau * mod)

    # ---- Continuous‑time update (Euler) ------------------------------------
    # We treat A = W @ x as the "driving" vector.
    A = W @ x_prev
    dx = -(1.0 / tau + mod) * x_prev + mod * A
    x_new = x_prev + dt * dx

    # ---- Simple surrogate loss and weight update ----------------------------
    # For demonstration we define a quadratic loss L = 0.5 * ||x_new - A||^2
    # Its gradient w.r.t. W is (W @ x_prev - x_new) ⊗ x_prev
    error = A - x_new
    grad_W = np.outer(error, x_prev)  # shape (n, n)

    # Learning rate scaled by fold‑change
    lr = eta * (1.0 + gamma * c_t)
    W_new = W - lr * grad_W

    return x_new, W_new, sig_curr


# ----------------------------------------------------------------------
# Public API – run hybrid dynamics over a sequence of texts
# ----------------------------------------------------------------------
def hybrid_forward(
    texts: List[str],
    inputs: List[float],
    hidden_dim: int = 16,
    tau: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    eta: float = 0.01,
    gamma: float = 0.2,
    dt: float = 0.1,
    num_perm: int = DEFAULT_NUM_PERM,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Execute the fused dynamics over a sequence.

    Parameters
    ----------
    texts : List[str]
        Token sequences (one per time step).
    inputs : List[float]
        External scalar input I_t for each time step.
    hidden_dim : int
        Dimensionality of the hidden state x_t.
    tau, alpha, beta, eta, gamma, dt : float
        Hyper‑parameters as described in the module docstring.
    num_perm : int
        Length of MinHash signatures.

    Returns
    -------
    states : List[np.ndarray]
        Hidden state after each step.
    weights : List[np.ndarray]
        Weight matrix after each step.
    """
    if len(texts) != len(inputs):
        raise ValueError("texts and inputs must have the same length")

    # Initialise hidden state, weight matrix, gating parameters, and MinHash cache
    x = np.zeros(hidden_dim, dtype=np.float64)
    W = np.eye(hidden_dim, dtype=np.float64)  # start with identity
    theta = np.random.randn(hidden_dim + 1) * 0.1  # small random init

    # Initialise MinHash signature with an empty token set
    sig_prev = np.full(num_perm, MAX64, dtype=np.uint64)

    states: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    I_prev = inputs[0]  # for the very first step fold_change will be zero

    for txt, I_curr in zip(texts, inputs):
        tokens = shingles(txt)
        x, W, sig_prev = hybrid_step(
            x_prev=x,
            I_curr=I_curr,
            I_prev=I_prev,
            W=W,
            theta=theta,
            tau=tau,
            alpha=alpha,
            beta=beta,
            eta=eta,
            gamma=gamma,
            dt=dt,
            sig_prev=sig_prev,
            tokens_curr=tokens,
        )
        states.append(x.copy())
        weights.append(W.copy())
        I_prev = I_curr

    return states, weights


# ----------------------------------------------------------------------
# Minimal VRAM‑slot‑plan stub (kept for compatibility with Parent B)
# ----------------------------------------------------------------------
def dummy_vram_plan(artifact_id: str, estimated_mb: int) -> dict:
    """Return a placeholder VRAM‑slot plan dictionary."""
    return {
        "artifact_id": artifact_id,
        "artifact_kind": "dummy",
        "action": "reserve",
        "estimated_mb": estimated_mb,
        "reason": "demo",
        "detail": {},
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic example
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog again",
        "a completely different sentence appears here",
        "the quick brown fox returns to the scene",
    ]
    # Random scalar inputs simulating e.g. sentiment scores or sensor readings
    random.seed(42)
    sample_inputs = [random.uniform(0.0, 1.0) for _ in sample_texts]

    states, weights = hybrid_forward(
        texts=sample_texts,
        inputs=sample_inputs,
        hidden_dim=8,
        tau=1.2,
        alpha=0.4,
        beta=0.2,
        eta=0.005,
        gamma=0.1,
        dt=0.05,
        num_perm=32,
    )

    print("Final hidden state:", states[-1])
    print("Final weight matrix (norm):", np.linalg.norm(weights[-1]))
    # Demonstrate dummy VRAM plan creation
    plan = dummy_vram_plan("demo_artifact", 128)
    print("Dummy VRAM plan:", plan)