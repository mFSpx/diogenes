# DARWIN HAMMER — match 39, survivor 1
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s2.py (gen1)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# born: 2026-05-29T23:23:40Z

"""
Hybrid Workshare-Calendar and Liquid-Time-Constant-MinHash algorithm.

This module fuses two parent algorithms:
* **Hybrid workshare-calendar allocator** – a deterministic work and LLM residuals 
  allocator that uses a weekday-dependent weight vector.
* **Hybrid Liquid-Time-Constant & MinHash Network** – a continuous-time recurrent 
  neural network whose effective time constant depends on a learned gating function 
  and a MinHash similarity between token sets.

The mathematical bridge is the integration of the weekday-dependent weight vector 
from the workshare-calendar allocator into the gating function of the Liquid-Time-Constant 
network, allowing it to modulate the effective liquid time constant based on both the 
learned gating and the MinHash similarity.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from datetime import date

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

# ----------------------------------------------------------------------
# Calendar helper
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: tuple, dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

# ----------------------------------------------------------------------
# MinHash utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    import hashlib
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: list, k: int = 128) -> list:
    """Return a MinHash signature of length *k* for the given token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def minhash_similarity(sig_a: list, sig_b: list) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def shingles(text: str, width: int = 5) -> set:
    """Extract width‑word shingles from *text*."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}

# ----------------------------------------------------------------------
# Hybrid step integrating MinHash similarity and weekday-dependent weight vector
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

def ltc_step_hybrid(
    x: np.ndarray,
    I: np.ndarray,
    prev_sig: list,
    cur_sig: list,
    params: dict,
    alpha: float = 0.5,
    dow: int = 0,
    dt: float = 0.1,
) -> tuple:
    """
    Perform one Euler integration step of the hybrid LTC‑MinHash dynamics.

    The effective gating is ``g = f + α·s_t`` where ``s_t`` is the MinHash 
    similarity between the previous and current signatures, modulated by the 
    weekday-dependent weight vector.
    """
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]
    weight_vec = weekday_weight_vector(GROUPS, dow)

    # Learned gating
    f_val = ltc_f(x, I, W, b)

    # MinHash similarity term
    s_t = minhash_similarity(prev_sig, cur_sig) if prev_sig else 0.0
    s_vec = np.full_like(f_val, s_t)

    # Combined gating modulated by weekday-dependent weight vector
    g = f_val + alpha * s_vec * weight_vec.sum()

    # ODE with combined gating
    dx_dt = -(1.0 / tau + g) * x + g * A

    x_new = x + dt * dx_dt

    # Effective time constant per neuron
    tau_eff_vec = tau / (1.0 + tau * g)
    tau_eff = float(np.mean(tau_eff_vec))

    return x_new, tau_eff, cur_sig

def hybrid_forward(
    texts: list,
    params: dict,
    alpha: float = 0.5,
    dow: int = 0,
    dt: float = 0.1,
) -> list:
    """
    Run the hybrid dynamics over a sequence of texts.
    """
    x = np.zeros((len(GROUPS),))
    prev_sig = None
    results = []
    for text in texts:
        tokens = shingles(text)
        cur_sig = minhash_signature(tokens)
        x_new, tau_eff, cur_sig = ltc_step_hybrid(x, np.zeros((len(GROUPS),)), prev_sig, cur_sig, params, alpha, dow, dt)
        results.append((x_new, tau_eff))
        x = x_new
        prev_sig = cur_sig
    return results

if __name__ == "__main__":
    params = {
        "W": np.random.rand(len(GROUPS), len(GROUPS) * 2),
        "b": np.zeros((len(GROUPS),)),
        "tau": 1.0,
        "A": np.random.rand(len(GROUPS),)
    }
    texts = ["This is a test", "This is another test"]
    results = hybrid_forward(texts, params)
    print(results)