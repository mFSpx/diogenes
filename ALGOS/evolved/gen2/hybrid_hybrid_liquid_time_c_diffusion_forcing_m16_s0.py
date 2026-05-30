# DARWIN HAMMER — match 16, survivor 0
# gen: 2
# parent_a: hybrid_liquid_time_constant_minhash_m10_s1.py (gen1)
# parent_b: diffusion_forcing.py (gen0)
# born: 2026-05-29T23:22:41Z

"""
Hybrid Liquid Time Constant Diffusion Forcing — a novel fusion of Liquid Time-Constant Networks 
(LTCs) and Diffusion Forcing (DF) for sequence planning and temporal pattern recognition. 

The mathematical bridge lies in integrating the Diffusion Forcing noise schedule within the LTC's 
input-dependent temporal dynamics, allowing the LTC to adaptively modulate its temporal response 
based on the noise level of the input sequence. This is achieved by modifying the LTC's network 
function to incorporate the Diffusion Forcing noise schedule as an additional input feature, 
effectively creating a feedback loop where the LTC's state influences the noise schedule and vice versa.

The Hybrid LTCDF architecture combines the strengths of both parents: the LTC's ability to learn 
complex patterns in sequential data and the DF's ability to condition on a mixture of clean and 
noisy tokens simultaneously.
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

def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, 1e-9, 1.0)
        return alpha_bars
    elif schedule == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.concatenate([[1.0], np.cumprod(alphas)])
        alpha_bars = np.clip(alpha_bars, 1e-9, 1.0)
        return alpha_bars
    else:
        raise ValueError(f"Unknown schedule '{schedule}'. Choose 'cosine' or 'linear'.")

def add_noise_token(x0_i: np.ndarray, t_i: int, alpha_bars: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ab = alpha_bars[t_i]
    epsilon = rng.standard_normal(x0_i.shape)
    x_ti = np.sqrt(ab) * x0_i + np.sqrt(1.0 - ab) * epsilon
    return x_ti, epsilon

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def shingles(text: str, width: int = 5) -> set[str]:
    words = text.split()
    if width <= 0:
        raise ValueError('width must be positive')
    if len(words) < width:
        return {' '.join(words)} if words else set()
    return {' '.join(words[i:i+width]) for i in range(len(words) - width + 1)}

def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray, sig: list[int], t: int, alpha_bars: np.ndarray) -> np.ndarray:
    concat = np.concatenate([x, I, np.array([similarity(sig, signature(shingles(" ".join(map(str, I)), k=5), k=5)), alpha_bars[t]])], axis=0)
    return sigmoid(W @ concat + b)

def ltc_step(x: np.ndarray, I: np.ndarray, params: dict, dt: float = 0.1, sig: list[int] = [], t: int = 0, alpha_bars: np.ndarray = None) -> tuple[np.ndarray, float, int]:
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    f_val = ltc_f(x, I, W, b, sig, t, alpha_bars)
    dx_dt = -(1.0 / tau + f_val) * x + f_val * A
    x_new = x + dt * dx_dt
    tau_sys_vec = tau / (1.0 + tau * f_val)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys, t + 1

def ltc_forward(I_seq: np.ndarray, params: dict, x0: np.ndarray | None = None, dt: float = 0.1, T: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_seq, input_dim = I_seq.shape
    hidden_dim = params["A"].shape[0]

    x = np.zeros(hidden_dim) if x0 is None else np.array(x0, dtype=float)

    X = np.empty((T_seq, hidden_dim))
    tau_sys_seq = np.empty(T_seq)
    t_seq = np.empty(T_seq, dtype=int)

    sig = []
    alpha_bars = noise_schedule(T)
    t = 0
    for i in range(T_seq):
        x, tau_sys, t = ltc_step(x, I_seq[i], params, dt=dt, sig=sig, t=t, alpha_bars=alpha_bars)
        X[i] = x
        tau_sys_seq[i] = tau_sys
        t_seq[i] = t
        sig = signature(shingles(" ".join(map(str, I_seq[:i+1])), k=5), k=5)

    return X, tau_sys_seq, t_seq

def init_ltc(hidden_dim: int, input_dim: int, tau: float = 1.0, seed: int = 0) -> dict:
    W = np.random.rand(hidden_dim, hidden_dim + input_dim + 2)
    b = np.random.rand(hidden_dim)
    A = np.random.rand(hidden_dim)

    return {"W": W, "b": b, "tau": tau, "A": A}

MAX64 = (1 << 64) - 1

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    params = init_ltc(10, 5)
    I_seq = np.random.rand(100, 5)
    X, tau_sys_seq, t_seq = ltc_forward(I_seq, params)
    print("LTCDF simulation completed successfully.")