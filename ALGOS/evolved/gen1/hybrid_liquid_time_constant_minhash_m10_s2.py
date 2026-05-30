# DARWIN HAMMER — match 10, survivor 2
# gen: 1
# parent_a: liquid_time_constant.py (gen0)
# parent_b: minhash.py (gen0)
# born: 2026-05-29T23:17:12Z

"""Hybrid Liquid-Time-Constant & MinHash Network.

This module fuses two parent algorithms:

* **Liquid Time‑Constant Networks (LTC)** – a continuous‑time recurrent neural
  network whose effective time constant τ_sys(t) depends on a learned gating
  function f(x(t),I(t),θ).

* **MinHash signatures** – a locality‑sensitive hashing scheme that yields an
  approximate Jaccard similarity between token sets.

**Mathematical bridge**

For an input text at discrete time *t* we first build a token set
`T_t` (shingles).  Its MinHash signature `σ_t` (length *k*) is obtained with the
standard MinHash construction.  The similarity with the previous signature
`σ_{t‑1}`,


s_t = Ĵ(σ_{t‑1}, σ_t) = similarity(σ_{t‑1}, σ_t) ∈ [0,1],


is used as an *extrinsic* scalar that modulates the effective liquid time
constant:


τ_eff(t) = τ / (1 + τ·(f(x_t, I_t, θ) + α·s_t)),


where `α ≥ 0` weighs the influence of the MinHash similarity.  The ODE becomes


dx/dt = -(1/τ + f + α·s_t)·x + (f + α·s_t)·A .


Thus the recurrent dynamics are driven jointly by the learned gating `f` and
the data‑dependent similarity `s_t`.  The hybrid network can react quickly
when the current text is similar to the previous one (large `s_t`) and
slowly otherwise, preserving the spirit of LTC while injecting a principled
set‑based similarity signal.

The implementation below provides:
* `minhash_signature` and `minhash_similarity` (parent B).
* `ltc_f` and `ltc_step_hybrid` (extended LTC step).
* `hybrid_forward` – runs the hybrid dynamics over a sequence of texts.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

__all__ = [
    "sigmoid",
    "ltc_f",
    "ltc_step_hybrid",
    "hybrid_forward",
    "minhash_signature",
    "minhash_similarity",
    "shingles",
]

# ----------------------------------------------------------------------
# Parent B – MinHash utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def shingles(text: str, width: int = 5) -> set[str]:
    """Extract width‑word shingles from *text*."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


# ----------------------------------------------------------------------
# Parent A – LTC primitives (slightly adapted)
# ----------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element‑wise sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gating function f(x,I) = σ(W·[x;I] + b)."""
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)


# ----------------------------------------------------------------------
# Hybrid step integrating MinHash similarity
# ----------------------------------------------------------------------


def ltc_step_hybrid(
    x: np.ndarray,
    I: np.ndarray,
    prev_sig: List[int],
    cur_sig: List[int],
    params: dict,
    alpha: float = 0.5,
    dt: float = 0.1,
) -> Tuple[np.ndarray, float, List[int]]:
    """
    Perform one Euler integration step of the hybrid LTC‑MinHash dynamics.

    The effective gating is ``g = f + α·s_t`` where ``s_t`` is the MinHash
    similarity between the previous and current signatures.

    Returns
    -------
    x_new : np.ndarray
        Updated hidden state.
    tau_eff : float
        Mean effective liquid time constant after the step.
    cur_sig : List[int]
        Current signature (to be used as ``prev_sig`` in the next step).
    """
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    # Learned gating
    f_val = ltc_f(x, I, W, b)  # (hidden_dim,)

    # MinHash similarity term
    s_t = minhash_similarity(prev_sig, cur_sig) if prev_sig else 0.0
    s_vec = np.full_like(f_val, s_t)  # broadcast scalar to each neuron

    # Combined gating
    g = f_val + alpha * s_vec  # (hidden_dim,)

    # ODE with combined gating
    dx_dt = -(1.0 / tau + g) * x + g * A

    x_new = x + dt * dx_dt

    # Effective time constant per neuron
    tau_eff_vec = tau / (1.0 + tau * g)
    tau_eff = float(np.mean(tau_eff_vec))

    return x_new, tau_eff, cur_sig


# ----------------------------------------------------------------------
# Hybrid forward pass over a list of texts
# ----------------------------------------------------------------------


def hybrid_forward(
    texts: List[str],
    params: dict,
    k: int = 128,
    alpha: float = 0.5,
    dt: float = 0.1,
    shingle_width: int = 5,
    x0: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Run the hybrid network on a sequence of raw texts.

    Parameters
    ----------
    texts : List[str]
        Input sequence (one document per time step).
    params : dict
        Same structure as LTC parameters (keys: "W","b","tau","A").
    k : int
        Length of MinHash signatures.
    alpha : float
        Weight of the MinHash similarity term.
    dt : float
        Euler integration step size.
    shingle_width : int
        Width of word shingles used to build token sets.
    x0 : np.ndarray | None
        Optional initial hidden state (zeros if None).

    Returns
    -------
    X : np.ndarray, shape (T, hidden_dim)
        Hidden state trajectory.
    tau_seq : np.ndarray, shape (T,)
        Effective time constant at each step.
    sig_seq : List[List[int]]
        MinHash signature for each input text.
    """
    T = len(texts)
    hidden_dim = params["A"].shape[0]

    x = np.zeros(hidden_dim) if x0 is None else np.array(x0, dtype=float)

    X = np.empty((T, hidden_dim))
    tau_seq = np.empty(T)
    sig_seq: List[List[int]] = []

    prev_sig: List[int] = []  # empty for the first step

    for t, txt in enumerate(texts):
        token_set = shingles(txt, width=shingle_width)
        cur_sig = minhash_signature(token_set, k=k)

        # Convert the (binary) token presence into a numeric input vector.
        # Here we simply use the normalized hash values as a proxy.
        I = np.array([h / MAX64 for h in cur_sig[: params["W"].shape[0] - hidden_dim]],
                     dtype=float)

        # Ensure I has the expected dimension (input_dim)
        input_dim = params["W"].shape[1] - hidden_dim
        if I.shape[0] < input_dim:
            I = np.pad(I, (0, input_dim - I.shape[0]), constant_values=0.0)
        elif I.shape[0] > input_dim:
            I = I[:input_dim]

        x, tau_eff, _ = ltc_step_hybrid(
            x, I, prev_sig, cur_sig, params, alpha=alpha, dt=dt
        )
        X[t] = x
        tau_seq[t] = tau_eff
        sig_seq.append(cur_sig)
        prev_sig = cur_sig

    return X, tau_seq, sig_seq


# ----------------------------------------------------------------------
# Simple parameter initialiser (mirrors parent A)
# ----------------------------------------------------------------------


def init_hybrid_params(
    hidden_dim: int,
    input_dim: int,
    tau: float = 1.0,
    seed: int = 0,
) -> dict:
    """Create a random parameter dictionary compatible with the hybrid network."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((hidden_dim, hidden_dim + input_dim))
    b = rng.standard_normal(hidden_dim)
    A = rng.standard_normal(hidden_dim)
    return {"W": W, "b": b, "tau": float(tau), "A": A}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox leaps over the lazy dog",
        "an entirely different sentence with other words",
        "the quick brown fox jumps over the lazy dog again",
    ]

    # Initialise hybrid model
    hidden = 16
    # Input dimension derived from signature length; we pick a modest size.
    input_dim = 32
    params = init_hybrid_params(hidden, input_dim, tau=1.5, seed=42)

    # Run forward pass
    X, tau_seq, sig_seq = hybrid_forward(
        corpus,
        params,
        k=64,
        alpha=0.8,
        dt=0.05,
        shingle_width=3,
        x0=None,
    )

    # Simple sanity prints
    print("Hidden states shape:", X.shape)
    print("Effective τ sequence:", tau_seq)
    print("First signature (truncated):", sig_seq[0][:5])
    # Verify that similarity influences τ (rough check)
    for i in range(1, len(corpus)):
        sim = minhash_similarity(sig_seq[i - 1], sig_seq[i])
        print(f"Step {i}: MinHash similarity = {sim:.3f}, τ_eff = {tau_seq[i]:.3f}")