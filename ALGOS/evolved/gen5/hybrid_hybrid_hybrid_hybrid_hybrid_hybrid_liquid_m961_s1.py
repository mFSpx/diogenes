# DARWIN HAMMER — match 961, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s2.py (gen4)
# parent_b: hybrid_hybrid_liquid_time_c_hybrid_hybrid_model__m158_s1.py (gen3)
# born: 2026-05-29T23:31:55Z

"""Hybrid TTT‑LTC‑Sketch Algorithm
================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s2.py*  
  Provides a ternary‑linear (TTT) weight matrix `W` that is updated by gradient
  descent, Laplace‑noise injection for differential privacy and a
  reconstruction‑risk ratio `r_t` derived from a Count‑Min sketch of
  quasi‑identifier strings.

* **Parent B** – *hybrid_hybrid_liquid_time_c_hybrid_hybrid_model__m158_s1.py*  
  Provides a Liquid‑Time‑Constant (LTC) recurrent dynamics whose effective
  time‑constant is modulated by three scalars:
  `f_t` (a sigmoid of the state and an external input),
  `s_t` (MinHash similarity between consecutive token sets) and
  `c_t` (fold‑change of the external input).

**Mathematical Bridge**

The bridge is built on the *scalar modulators* that appear in both parents.
We construct a single scalar


m_t = f_t + α·s_t + β·r_t + γ·c_t


and use it

* to shrink the LTC effective time‑constant `τ_eff(t) = τ / (1 + τ·m_t)`,
* as the coefficient that mixes the autonomous decay and the driven term
  `W_t @ x_t` in the state update,
* and as a multiplicative factor for the learning‑rate of the TTT weight
  matrix.

Thus the TTT weight matrix becomes the recurrent weight matrix of the LTC
system, while the privacy‑preserving Count‑Min sketch supplies the additional
modulator `r_t`.  The resulting unified dynamics are


x_{t+1} = x_t + Δt·[ -(1/τ + m_t)·x_t + m_t·(W_t @ x_t) ]
W_{t+1} = W_t - η·(1 + λ·r_t)·∂L/∂W_t + Laplace(σ)


where `L` is the TTT reconstruction loss and `λ` controls how much the
privacy‑risk ratio influences the learning step.

The module implements the full pipeline and provides three public functions
that demonstrate the hybrid operation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyper‑parameters (can be tuned by the user)
# ----------------------------------------------------------------------
TAU = 1.0            # base time constant for LTC
ALPHA = 0.5          # weight of MinHash similarity
BETA = 0.3           # weight of reconstruction‑risk ratio
GAMMA = 0.2          # weight of fold‑change term
LAMBDA = 0.1         # privacy‑risk scaling for weight update
ETA = 0.01           # base learning rate
DT = 0.1             # Euler integration step
LAPLACE_SCALE = 1e-3 # scale of Laplace noise injected into W
EPS = 1e-9           # small constant to avoid division by zero

# ----------------------------------------------------------------------
# Helper utilities – Parent A (TTT + Count‑Min sketch + privacy)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01,
             seed: int = 0) -> np.ndarray:
    """Initialize the TTT‑Linear weight matrix."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in), dtype=np.float64) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """Quadratic reconstruction loss used by the TTT component."""
    if target is None:
        target = x
    return float(np.sum((W @ x - target) ** 2))


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Gradient of the TTT loss w.r.t. the weight matrix."""
    if target is None:
        target = x
    # ∂/∂W ½‖W x - target‖² = (W x - target) xᵀ
    diff = (W @ x - target).reshape(-1, 1)          # (d_out, 1)
    grad = 2.0 * diff @ x.reshape(1, -1)            # (d_out, d_in)
    return grad


def laplace_noise(shape: Tuple[int, int], scale: float = LAPLACE_SCALE) -> np.ndarray:
    """Draw Laplace noise (zero‑mean) for differential privacy."""
    rng = np.random.default_rng()
    u = rng.uniform(-0.5, 0.5, size=shape)
    return -scale * np.sign(u) * np.log(1 - 2 * np.abs(u))


# Count‑Min sketch -------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """64‑bit hash of a token with a 4‑byte seed using Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


class CountMinSketch:
    """Simple Count‑Min sketch with Laplace‑noise‑aware queries."""

    def __init__(self, depth: int = 8, width: int = 256, seed: int = 0):
        self.depth = depth
        self.width = width
        self.seed = seed
        self.table = np.zeros((depth, width), dtype=np.float64)
        # Pre‑compute hash seeds for each row
        self.row_seeds = [(seed + i) & 0xFFFFFFFF for i in range(depth)]

    def update(self, tokens: Iterable[str]) -> None:
        """Increment counters for each token."""
        for token in tokens:
            for r, row_seed in enumerate(self.row_seeds):
                idx = _hash(row_seed, token) % self.width
                self.table[r, idx] += 1.0

    def query(self, token: str) -> float:
        """Estimate the count of a token (minimum over rows)."""
        estimates = []
        for r, row_seed in enumerate(self.row_seeds):
            idx = _hash(row_seed, token) % self.width
            estimates.append(self.table[r, idx])
        return float(min(estimates))

    def noisy_query(self, token: str, scale: float = LAPLACE_SCALE) -> float:
        """Privacy‑preserving query with Laplace noise."""
        return self.query(token) + np.random.laplace(0.0, scale)


def reconstruction_risk_ratio(W: np.ndarray, sketch: CountMinSketch,
                              tokens: Iterable[str], x: np.ndarray) -> float:
    """
    Compute a scalar that measures how well the current weight matrix
    reconstructs the sketch‑derived counts.
    The ratio is defined as

        r = ||W x - μ|| / (||μ|| + ε)

    where μ is the vector of (noisy) sketch counts for the supplied tokens.
    """
    mu = np.array([sketch.noisy_query(tok) for tok in tokens], dtype=np.float64)
    if mu.size == 0:
        return 0.0
    recon = W @ x
    numer = np.linalg.norm(recon - mu)
    denom = np.linalg.norm(mu) + EPS
    return float(numer / denom)


# ----------------------------------------------------------------------
# Helper utilities – Parent B (LTC + MinHash + Fold‑Change)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
DEFAULT_NUM_PERM = 64  # length of MinHash signature


def _minhash_signature(tokens: Iterable[str], num_perm: int = DEFAULT_NUM_PERM,
                       seed: int = 0) -> np.ndarray:
    """Compute a MinHash signature (uint64 array) for a set of tokens."""
    rng = np.random.default_rng(seed)
    # Generate random hash seeds for each permutation
    perm_seeds = rng.integers(0, MAX64, size=num_perm, dtype=np.uint64)
    signature = np.full(num_perm, MAX64, dtype=np.uint64)

    for token in tokens:
        token_bytes = token.encode("utf-8", errors="ignore")
        token_hash = int.from_bytes(hashlib.blake2b(token_bytes, digest_size=8).digest(),
                                    "big")
        for i, p in enumerate(perm_seeds):
            combined = (token_hash ^ p) & MAX64
            if combined < signature[i]:
                signature[i] = combined
    return signature


def minhash_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("Signatures must have the same length")
    return float(np.mean(sig_a == sig_b))


def fold_change(I_t: float, I_prev: float) -> float:
    """Fold‑change scalar c_t = (I_t - I_{t-1}) / (|I_{t-1}| + ε)."""
    return (I_t - I_prev) / (abs(I_prev) + EPS)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element‑wise sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------
# Core hybrid state and dynamics
# ----------------------------------------------------------------------
class HybridState:
    """Container for all mutable objects of the hybrid system."""

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 sketch_depth: int = 8,
                 sketch_width: int = 256,
                 seed: int = 0):
        self.W = init_ttt(d_in, d_out, seed=seed)               # TTT weight matrix
        self.sketch = CountMinSketch(depth=sketch_depth,
                                     width=sketch_width,
                                     seed=seed)
        self.prev_tokens: List[str] = []                        # for MinHash similarity
        self.prev_I: float = 0.0                                # previous external input
        self.prev_sig: np.ndarray | None = None                # previous MinHash signature


def init_hybrid(d_in: int,
                d_out: int,
                sketch_depth: int = 8,
                sketch_width: int = 256,
                seed: int = 0) -> HybridState:
    """Factory that creates a fresh HybridState."""
    return HybridState(d_in, d_out, sketch_depth, sketch_width, seed)


def hybrid_step(state: HybridState,
                x_t: np.ndarray,
                I_t: float,
                tokens: Iterable[str]) -> Tuple[np.ndarray, float]:
    """
    Perform one discrete time‑step of the fused system.

    Parameters
    ----------
    state : HybridState
        Current mutable state.
    x_t : np.ndarray (d_in,)
        Current state vector.
    I_t : float
        External scalar input.
    tokens : iterable of str
        Current set of quasi‑identifier strings (e.g. tokenised text).

    Returns
    -------
    x_next : np.ndarray
        Updated state vector.
    loss   : float
        TTT reconstruction loss after the update.
    """
    # ------------------------------------------------------------------
    # 1. Update the Count‑Min sketch with the new tokens
    # ------------------------------------------------------------------
    token_list = list(tokens)
    state.sketch.update(token_list)

    # ------------------------------------------------------------------
    # 2. Compute scalar modulators
    # ------------------------------------------------------------------
    # a) f_t from sigmoid(θ·[x;I]) – we use a simple linear projection θ = 1
    concat = np.concatenate([x_t, np.array([I_t], dtype=np.float64)])
    f_t = float(sigmoid(concat).mean())  # scalar approximation

    # b) MinHash similarity s_t with previous token set
    cur_sig = _minhash_signature(token_list, seed=state.sketch.seed)
    if state.prev_sig is None:
        s_t = 0.0
    else:
        s_t = minhash_similarity(state.prev_sig, cur_sig)

    # c) Fold‑change c_t of the external input
    c_t = fold_change(I_t, state.prev_I)

    # d) Reconstruction‑risk ratio r_t from TTT + sketch
    r_t = reconstruction_risk_ratio(state.W, state.sketch, token_list, x_t)

    # Combined modulator
    m_t = f_t + ALPHA * s_t + BETA * r_t + GAMMA * c_t

    # ------------------------------------------------------------------
    # 3. LTC‑style state update (Euler discretisation)
    # ------------------------------------------------------------------
    decay = (1.0 / TAU) + m_t
    drive = m_t
    x_next = x_t + DT * ( -decay * x_t + drive * (state.W @ x_t) )

    # ------------------------------------------------------------------
    # 4. Weight matrix update using TTT gradient and privacy‑aware LR
    # ------------------------------------------------------------------
    loss = ttt_loss(state.W, x_t)
    grad = ttt_grad(state.W, x_t)
    lr = ETA * (1.0 + LAMBDA * r_t)   # privacy‑scaled learning rate
    state.W -= lr * grad
    # Inject Laplace noise for differential privacy
    state.W += laplace_noise(state.W.shape)

    # ------------------------------------------------------------------
    # 5. Store values for the next step
    # ------------------------------------------------------------------
    state.prev_tokens = token_list
    state.prev_I = I_t
    state.prev_sig = cur_sig

    return x_next, loss


def hybrid_evaluate(state: HybridState,
                    x: np.ndarray,
                    tokens: Iterable[str]) -> Dict[str, float]:
    """
    Compute a collection of diagnostic metrics for the current state.

    Returns a dictionary containing:
        - 'ttt_loss'   : reconstruction loss
        - 'risk_ratio' : reconstruction‑risk ratio r_t
        - 'minhash_sim': similarity with previous token set (0 if none)
        - 'fold_change': last fold‑change scalar c_t
    """
    token_list = list(tokens)

    # loss
    loss = ttt_loss(state.W, x)

    # risk ratio
    r = reconstruction_risk_ratio(state.W, state.sketch, token_list, x)

    # minhash similarity
    cur_sig = _minhash_signature(token_list, seed=state.sketch.seed)
    if state.prev_sig is None:
        sim = 0.0
    else:
        sim = minhash_similarity(state.prev_sig, cur_sig)

    # fold change (uses stored previous I)