#!/usr/bin/env python3
"""
Diffusion Forcing — reference implementation for sequence planning.

Standard diffusion assigns a single noise level t to the entire sequence at each
training step. Every token is corrupted uniformly: x_t = sqrt(alpha_bar_t) * x_0
+ sqrt(1 - alpha_bar_t) * epsilon.

Diffusion Forcing (Chen et al. 2024) assigns an independent noise level t_i to
each token i in the sequence of length N. This lets the model see a mixture of
clean and noisy tokens simultaneously, enabling:

  - Causal planning: condition on a clean prefix (t_i = 0) while predicting a
    noisy suffix (t_i ~ Uniform[1, T]).
  - Flexible time horizons: near-future tokens at lower noise, far-future at
    higher noise, within a single forward pass.
  - Training as a byproduct of an independent-noise corruption process, which
    is compatible with standard DDPM-style objectives.

Loss equation:

    L_DF = E_{t_{1:N}, x_0, epsilon} [
        sum_{i=1}^{N} lambda(t_i)
            || epsilon^(i) - epsilon_theta(
                   x_{t_1}^(1), ..., x_{t_N}^(N), t_{1:N}
               )^(i) ||^2
    ]

where lambda(t) is a per-timestep weighting function (e.g. SNR-based).

The key insight is that at planning time, known past tokens are passed at t=0
(clean) while unknown future tokens are passed at t=T (pure noise), giving the
denoiser a causally correct conditioning signal without masking or autoregressive
rollout overhead.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "noise_schedule",
    "add_noise_token",
    "add_noise_sequence",
    "weighting_lambda",
    "diffusion_forcing_loss",
    "sample_causal_t_seq",
]


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Return the cumulative noise schedule alpha_bar, shape (T+1,).

    alpha_bar[0] = 1.0  (clean)
    alpha_bar[T] ~ 0.0  (pure noise)

    Parameters
    ----------
    T:
        Total number of diffusion timesteps.
    schedule:
        "cosine" (Nichol & Dhariwal 2021) or "linear" (Ho et al. 2020).

    Returns
    -------
    np.ndarray shape (T+1,) with values in (0, 1].
    """
    if schedule == "cosine":
        # alpha_bar_t = cos^2( (t/T + s) / (1 + s) * pi/2 ) / cos^2( s / (1+s) * pi/2 )
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        # Clip to ensure numerical stability — never exactly zero.
        alpha_bars = np.clip(alpha_bars, 1e-9, 1.0)
        return alpha_bars
    elif schedule == "linear":
        # Beta linearly spaced from beta_start to beta_end; alpha_bar = prod(1 - beta).
        beta_start = 1e-4
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.concatenate([[1.0], np.cumprod(alphas)])
        alpha_bars = np.clip(alpha_bars, 1e-9, 1.0)
        return alpha_bars
    else:
        raise ValueError(f"Unknown schedule '{schedule}'. Choose 'cosine' or 'linear'.")


def add_noise_token(
    x0_i: np.ndarray,
    t_i: int,
    alpha_bars: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a single token x0_i to noise level t_i.

    Forward process (reparametrised):
        x_{t_i}^(i) = sqrt(alpha_bar[t_i]) * x0_i
                     + sqrt(1 - alpha_bar[t_i]) * epsilon

    Parameters
    ----------
    x0_i:
        Clean token, shape (d,).
    t_i:
        Integer timestep in [0, T].
    alpha_bars:
        Cumulative schedule from noise_schedule(), shape (T+1,).
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    (x_ti, epsilon) both shape (d,).
    """
    ab = alpha_bars[t_i]
    epsilon = rng.standard_normal(x0_i.shape)
    x_ti = np.sqrt(ab) * x0_i + np.sqrt(1.0 - ab) * epsilon
    return x_ti, epsilon


def add_noise_sequence(
    x0: np.ndarray,
    t_seq: np.ndarray,
    alpha_bars: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt every token in a sequence with its own independent noise level.

    Parameters
    ----------
    x0:
        Clean sequence, shape (N, d).
    t_seq:
        Per-token timesteps, shape (N,), integer values in [0, T].
    alpha_bars:
        Cumulative schedule, shape (T+1,).
    rng:
        NumPy random generator.

    Returns
    -------
    (x_noisy shape (N, d), epsilon shape (N, d))
    """
    N, d = x0.shape
    x_noisy = np.empty_like(x0)
    epsilon = np.empty_like(x0)
    for i in range(N):
        x_noisy[i], epsilon[i] = add_noise_token(x0[i], int(t_seq[i]), alpha_bars, rng)
    return x_noisy, epsilon


def weighting_lambda(t: int, T: int, alpha_bars: np.ndarray | None = None) -> float:
    """Per-timestep loss weighting lambda(t).

    Uses SNR weighting: lambda(t) = alpha_bar[t] / (1 - alpha_bar[t]),
    clipped to [0.01, 100] for numerical stability.

    At t=0 (clean token) SNR is very high; at t=T (pure noise) SNR ~ 0.
    Clipping prevents the loss from exploding at either extreme.

    Parameters
    ----------
    t:
        Timestep integer in [0, T].
    T:
        Total diffusion steps (used to build a cosine schedule if alpha_bars
        is not supplied).
    alpha_bars:
        Precomputed schedule shape (T+1,). Computed on the fly if None.

    Returns
    -------
    float scalar weight.
    """
    if alpha_bars is None:
        alpha_bars = noise_schedule(T, schedule="cosine")
    ab = float(np.clip(alpha_bars[t], 0.0, 1.0 - 1e-9))
    snr = ab / (1.0 - ab)
    return float(np.clip(snr, 0.01, 100.0))


def diffusion_forcing_loss(
    x0: np.ndarray,
    eps_pred: np.ndarray,
    t_seq: np.ndarray,
    alpha_bars: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Compute the Diffusion Forcing loss L_DF for one example.

    L_DF = sum_{i=1}^{N} lambda(t_i) * || epsilon^(i) - eps_pred^(i) ||^2

    The true epsilon values are sampled fresh inside this function (the model's
    prediction eps_pred is provided externally, typically from a neural denoiser).

    Parameters
    ----------
    x0:
        Clean sequence, shape (N, d).
    eps_pred:
        Denoiser prediction for the noise, shape (N, d).
    t_seq:
        Per-token timesteps, shape (N,), integer values in [0, T].
    alpha_bars:
        Cumulative schedule, shape (T+1,).
    rng:
        NumPy random generator used to draw the ground-truth noise.

    Returns
    -------
    Scalar float loss value.
    """
    T = len(alpha_bars) - 1
    _x_noisy, epsilon_true = add_noise_sequence(x0, t_seq, alpha_bars, rng)
    loss = 0.0
    for i in range(x0.shape[0]):
        lam = weighting_lambda(int(t_seq[i]), T, alpha_bars)
        diff = epsilon_true[i] - eps_pred[i]
        loss += lam * float(np.dot(diff, diff))
    return loss


def sample_causal_t_seq(
    N: int,
    T: int,
    clean_prefix: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a per-token noise-level sequence for causal planning.

    Planning mode assigns:
      - t_i = 0  for i in [0, clean_prefix)   → clean past (conditioning)
      - t_i ~ Uniform[1, T]  for i in [clean_prefix, N)  → noisy future (prediction)

    This is the core Diffusion Forcing trick: pass observed history at zero
    noise so the denoiser sees it cleanly, while corrupting the unobserved
    future uniformly so the model must predict all horizons simultaneously.

    Parameters
    ----------
    N:
        Sequence length.
    T:
        Maximum timestep (inclusive upper bound for noisy tokens).
    clean_prefix:
        Number of leading tokens forced to t=0. Must be in [0, N].
    rng:
        NumPy random generator. Created fresh if None.

    Returns
    -------
    np.ndarray shape (N,) dtype int64.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0 <= clean_prefix <= N):
        raise ValueError(f"clean_prefix={clean_prefix} must be in [0, {N}].")
    t_seq = np.zeros(N, dtype=np.int64)
    n_noisy = N - clean_prefix
    if n_noisy > 0:
        t_seq[clean_prefix:] = rng.integers(1, T + 1, size=n_noisy)
    return t_seq


if __name__ == "__main__":
    # ------------------------------------------------------------------ demo
    # Sequence planning: N=10 tokens, d=4 features, T=1000 diffusion steps.
    # Clean prefix of 3 tokens (observed past), noisy suffix of 7 (future).
    # ------------------------------------------------------------------ demo

    N = 10
    d = 4
    T = 1000
    clean_prefix = 3

    rng = np.random.default_rng(seed=42)

    alpha_bars = noise_schedule(T, schedule="cosine")
    x0 = rng.standard_normal((N, d))
    t_seq = sample_causal_t_seq(N, T, clean_prefix=clean_prefix, rng=rng)

    # Simulate a denoiser prediction (random stand-in for a neural network).
    eps_pred = rng.standard_normal((N, d))

    loss = diffusion_forcing_loss(x0, eps_pred, t_seq, alpha_bars, rng)

    print("Diffusion Forcing — causal planning demo")
    print(f"  N={N}  d={d}  T={T}  clean_prefix={clean_prefix}")
    print(f"  t_seq : {t_seq.tolist()}")
    print(f"  L_DF  : {loss:.6f}")
