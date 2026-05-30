# DARWIN HAMMER — match 10, survivor 1
# gen: 1
# parent_a: liquid_time_constant.py (gen0)
# parent_b: minhash.py (gen0)
# born: 2026-05-29T23:17:12Z

#!/usr/bin/env python3
"""
Hybrid Liquid Time Constant MinHash (LTCMH) — a novel fusion of Liquid Time-Constant Networks (LTCs)
and MinHash signatures for approximate Jaccard similarity. The mathematical bridge lies in integrating
the MinHash signature generation process within the LTC's input-dependent temporal dynamics. This is
achieved by modifying the LTC's network function to incorporate the MinHash signature similarity as
an additional input feature, effectively creating a feedback loop where the LTC's state influences
the MinHash signature generation and vice versa.

The LTCMH architecture combines the strengths of both parents: the LTC's ability to adaptively modulate
its temporal response based on the input, and the MinHash signature's efficient computation of approximate
Jaccard similarity. The fusion enables the LTCMH to learn complex patterns in sequential data while
incorporating a notion of similarity between the input sequences.
"""

import numpy as np
import hashlib
import math
import random
import sys
import pathlib

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def shingles(text: str, width: int = 5) -> set[str]:
    words = text.split()
    if width <= 0:
        raise ValueError('width must be positive')
    if len(words) < width:
        return {' '.join(words)} if words else set()
    return {' '.join(words[i:i+width]) for i in range(len(words) - width + 1)}

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray, sig: list[int]) -> np.ndarray:
    concat = np.concatenate([x, I, np.array([similarity(sig, signature(shingles(" ".join(map(str, I)), k=5), k=5))])], axis=0)
    return sigmoid(W @ concat + b)

def ltc_step(x: np.ndarray, I: np.ndarray, params: dict, dt: float = 0.1, sig: list[int] = []) -> tuple[np.ndarray, float]:
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    f_val = ltc_f(x, I, W, b, sig)
    dx_dt = -(1.0 / tau + f_val) * x + f_val * A
    x_new = x + dt * dx_dt
    tau_sys_vec = tau / (1.0 + tau * f_val)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys

def ltc_forward(I_seq: np.ndarray, params: dict, x0: np.ndarray | None = None, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    T, input_dim = I_seq.shape
    hidden_dim = params["A"].shape[0]

    x = np.zeros(hidden_dim) if x0 is None else np.array(x0, dtype=float)

    X = np.empty((T, hidden_dim))
    tau_sys_seq = np.empty(T)

    sig = []
    for t in range(T):
        x, tau_sys = ltc_step(x, I_seq[t], params, dt=dt, sig=sig)
        X[t] = x
        tau_sys_seq[t] = tau_sys
        sig = signature(shingles(" ".join(map(str, I_seq[:t+1])), k=5), k=5)

    return X, tau_sys_seq

def init_ltc(hidden_dim: int, input_dim: int, tau: float = 1.0, seed: int = 0) -> dict:
    W = np.random.rand(hidden_dim, hidden_dim + input_dim + 1)
    b = np.random.rand(hidden_dim)
    A = np.random.rand(hidden_dim)

    return {"W": W, "b": b, "tau": tau, "A": A}

MAX64 = (1 << 64) - 1

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    params = init_ltc(10, 5)
    I_seq = np.random.rand(100, 5)
    X, tau_sys_seq = ltc_forward(I_seq, params)
    print("LTCMH simulation completed successfully.")