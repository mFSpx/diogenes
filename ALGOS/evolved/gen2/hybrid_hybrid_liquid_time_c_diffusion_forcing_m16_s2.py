# DARWIN HAMMER — match 16, survivor 2
# gen: 2
# parent_a: hybrid_liquid_time_constant_minhash_m10_s1.py (gen1)
# parent_b: diffusion_forcing.py (gen0)
# born: 2026-05-29T23:22:41Z

"""Hybrid Liquid Time Constant Diffusion Forcing (LTC-DF)

This module fuses the core mathematics of two parent algorithms:

* **Parent A – `hybrid_liquid_time_constant_minhash_m10_s1.py`**  
  Provides a Liquid Time‑Constant (LTC) recurrent cell whose dynamics depend on an
  input‑dependent similarity term derived from MinHash signatures.

* **Parent B – `diffusion_forcing.py`**  
  Implements per‑token diffusion forcing where each token of a sequence is corrupted
  with an independent noise level `t_i`. The loss is weighted by a signal‑to‑noise
  ratio λ(t).

**Mathematical bridge**  
At every LTC step we compute a MinHash similarity `s ∈ [0,1]` between the current
input‐shingle signature and the accumulated signature so far. This similarity is
translated into a diffusion timestep


t_i = round( (1 - s) * T )


for each token (dimension) of the current input vector. Thus the LTC state governs
the amount of noise injected by the diffusion process, while the noisy input
returned to the LTC influences the next signature, closing a feedback loop.

The hybrid system therefore evolves according to


f(x, I, τ, A, s) = σ( W·[x; I; s] + b )
dx/dt          = -(1/τ + f)·x + f·A
t_i            = round( (1 - s) * T )
x_noisy_i      = √ᾱ[t_i]·I_i + √(1-ᾱ[t_i])·ε_i


where `σ` is the sigmoid, `ᾱ` the cumulative diffusion schedule, and `ε_i` standard
Gaussian noise. The loss aggregates the per‑token diffusion forcing loss weighted
by λ(t_i).

The implementation below contains three public functions that demonstrate the
hybrid operation and a smoke test.
"""

import numpy as np
import hashlib
import math
import random
import sys
import pathlib
from typing import Tuple, List

# ----------------------------------------------------------------------
# MinHash utilities (from Parent A)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def shingles(text: str, width: int = 5) -> List[str]:
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + width]) for i in range(len(words) - width + 1)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(x, dtype=float)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


# ----------------------------------------------------------------------
# Diffusion forcing utilities (from Parent B)
# ----------------------------------------------------------------------


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


def weighting_lambda(t: int, T: int, alpha_bars: np.ndarray | None = None) -> float:
    if alpha_bars is None:
        alpha_bars = noise_schedule(T, schedule="cosine")
    ab = float(np.clip(alpha_bars[t], 0.0, 1.0 - 1e-9))
    snr = ab / (1.0 - ab)
    return float(np.clip(snr, 0.01, 100.0))


def add_noise_token(
    x0_i: np.ndarray,
    t_i: int,
    alpha_bars: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    ab = alpha_bars[t_i]
    epsilon = rng.standard_normal(x0_i.shape)
    x_ti = np.sqrt(ab) * x0_i + np.sqrt(1.0 - ab) * epsilon
    return x_ti, epsilon


def add_noise_sequence(
    x0: np.ndarray,
    t_seq: np.ndarray,
    alpha_bars: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    N, d = x0.shape
    x_noisy = np.empty_like(x0)
    epsilon = np.empty_like(x0)
    for i in range(N):
        x_noisy[i], epsilon[i] = add_noise_token(
            x0[i], int(t_seq[i]), alpha_bars, rng
        )
    return x_noisy, epsilon


def diffusion_forcing_loss(
    eps_true: np.ndarray,
    eps_pred: np.ndarray,
    t_seq: np.ndarray,
    T: int,
    alpha_bars: np.ndarray,
) -> float:
    """Compute L_DF = Σ_i λ(t_i)·||ε_i - ε̂_i||²."""
    N = eps_true.shape[0]
    loss = 0.0
    for i in range(N):
        lam = weighting_lambda(int(t_seq[i]), T, alpha_bars)
        diff = eps_true[i] - eps_pred[i]
        loss += lam * np.sum(diff * diff)
    return loss / N


# ----------------------------------------------------------------------
# Hybrid LTC‑Diffusion core
# ----------------------------------------------------------------------


def init_hybrid_params(
    hidden_dim: int, input_dim: int, tau: float = 1.0, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    W = rng.random((hidden_dim, hidden_dim + input_dim + 1))
    b = rng.random(hidden_dim)
    A = rng.random(hidden_dim)
    return {"W": W, "b": b, "tau": tau, "A": A}


def hybrid_ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: dict,
    sig: List[int],
    dt: float = 0.1,
) -> Tuple[np.ndarray, float, float]:
    """
    One LTC integration step that also returns the current MinHash similarity.

    Returns
    -------
    x_new : updated hidden state
    tau_sys : effective time constant after the step
    sim : MinHash similarity used as an extra input feature
    """
    W, b, tau, A = params["W"], params["b"], float(params["tau"]), params["A"]

    # Compute similarity between current shingle signature and accumulated signature
    cur_sig = signature(shingles(" ".join(map(str, I.tolist())), k=5), k=5)
    sim = similarity(sig, cur_sig) if sig else 0.0

    concat = np.concatenate([x, I, np.array([sim])])
    f_val = sigmoid(W @ concat + b)

    dx_dt = -(1.0 / tau + f_val) * x + f_val * A
    x_new = x + dt * dx_dt

    tau_sys_vec = tau / (1.0 + tau * f_val)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys, sim


def hybrid_forward(
    I_seq: np.ndarray,
    params: dict,
    T_diff: int,
    rng: np.random.Generator,
    dt: float = 0.1,
    schedule: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the hybrid LTC‑Diffusion system over an input sequence.

    Returns
    -------
    X          : hidden states (T, hidden_dim)
    noisy_seq  : noisy version of the input sequence after diffusion forcing
    tau_seq    : effective time constants per step
    """
    T, input_dim = I_seq.shape
    hidden_dim = params["A"].shape[0]

    x = np.zeros(hidden_dim)
    X = np.empty((T, hidden_dim))
    tau_seq = np.empty(T)

    # Initial empty signature (all max values)
    sig = [MAX64] * 5

    # Pre‑compute diffusion schedule
    alpha_bars = noise_schedule(T_diff, schedule=schedule)

    # We'll build a per‑step token‑wise timestep vector based on similarity
    t_seq = np.empty(T, dtype=int)

    for t in range(T):
        I_t = I_seq[t]
        x, tau_sys, sim = hybrid_ltc_step(x, I_t, params, sig, dt=dt)

        # Translate similarity into a diffusion timestep for this token
        t_i = int(round((1.0 - sim) * T_diff))
        t_i = int(np.clip(t_i, 0, T_diff))
        t_seq[t] = t_i

        # Update signature with the new shingle (using cumulative input up to now)
        cumulative_text = " ".join(map(str, I_seq[: t + 1].reshape(-1).tolist()))
        sig = signature(shingles(cumulative_text, k=5), k=5)

        X[t] = x
        tau_seq[t] = tau_sys

    # Apply diffusion forcing to the whole input sequence using the computed timesteps
    noisy_seq, _ = add_noise_sequence(I_seq, t_seq, alpha_bars, rng)

    return X, noisy_seq, tau_seq


def hybrid_loss(
    I_seq: np.ndarray,
    params: dict,
    T_diff: int,
    rng: np.random.Generator,
    dt: float = 0.1,
    schedule: str = "cosine",
) -> float:
    """
    Compute the diffusion‑forcing loss for a single example, where the per‑token
    timesteps are generated by the LTC‑MinHash dynamics.
    """
    _, noisy_seq, _ = hybrid_forward(I_seq, params, T_diff, rng, dt=dt, schedule=schedule)

    # For the purpose of loss we assume a perfect denoiser that predicts the true epsilon.
    # We therefore reconstruct epsilon from the forward corruption formula.
    alpha_bars = noise_schedule(T_diff, schedule=schedule)

    N, d = I_seq.shape
    epsilon_true = np.empty_like(I_seq)
    t_seq = np.empty(N, dtype=int)

    # Recover the timesteps used during forward (by re‑computing similarity)
    sig = [MAX64] * 5
    for i in range(N):
        cumulative_text = " ".join(map(str, I_seq[: i + 1].reshape(-1).tolist()))
        cur_sig = signature(shingles(cumulative_text, k=5), k=5)
        sim = similarity(sig, cur_sig) if sig else 0.0
        t_i = int(round((1.0 - sim) * T_diff))
        t_i = int(np.clip(t_i, 0, T_diff))
        t_seq[i] = t_i
        sig = cur_sig

        ab = alpha_bars[t_i]
        eps = (noisy_seq[i] - np.sqrt(ab) * I_seq[i]) / np.sqrt(1.0 - ab)
        epsilon_true[i] = eps

    # Dummy predictor returns zeros (worst case) to illustrate loss computation
    eps_pred = np.zeros_like(epsilon_true)

    loss = diffusion_forcing_loss(epsilon_true, eps_pred, t_seq, T_diff, alpha_bars)
    return loss


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    hidden_dim = 8
    input_dim = 4
    T_seq = 20
    T_diff = 10

    params = init_hybrid_params(hidden_dim, input_dim, tau=1.2, seed=123)
    I_seq = rng.random((T_seq, input_dim))

    X, noisy_seq, tau_seq = hybrid_forward(I_seq, params, T_diff, rng)
    print("Hidden states shape:", X.shape)
    print("Noisy sequence shape:", noisy_seq.shape)
    print("Effective tau stats:", tau_seq.min(), tau_seq.max())

    loss_val = hybrid_loss(I_seq, params, T_diff, rng)
    print("Hybrid diffusion‑forcing loss:", loss_val)