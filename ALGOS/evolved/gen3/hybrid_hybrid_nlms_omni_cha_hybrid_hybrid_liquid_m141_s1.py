# DARWIN HAMMER — match 141, survivor 1
# gen: 3
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s1.py (gen1)
# parent_b: hybrid_hybrid_liquid_time_c_diffusion_forcing_m16_s0.py (gen2)
# born: 2026-05-29T23:25:48Z

"""Hybrid NLMS‑LTC Diffusion Fusion

This module fuses two distinct parent algorithms:

* **Parent A** – a Normalized Least‑Mean‑Squares (NLMS) adaptive filter.
* **Parent B** – a Liquid‑Time‑Constant (LTC) diffusion‑forcing schedule with
  MinHash‑based token signatures.

The mathematical bridge is the element‑wise scaling of the diffusion schedule
vector  α̅ₜ  by a weight vector **w** that is continuously adapted with the NLMS
update rule.  The NLMS filter receives a feature vector derived from the MinHash
signature of the current token set, treats the scaled schedule as the linear
model output, and updates **w** to minimise the prediction error.  Thus the
temporal diffusion dynamics become learnable and data‑driven.

The core hybrid operations are:
1. `nlms_update` – classic NLMS weight adaptation.
2. `noise_schedule` – deterministic diffusion schedule (cosine or linear).
3. `hybrid_predict` – prediction using the scaled schedule and signature‑derived
   features.
4. `hybrid_train` – one‑pass training loop that ties the two components together.
"""

import sys
import math
import random
import hashlib
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Parent B – MinHash signature utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: list[str], k: int = 128) -> list[int]:
    """MinHash signature of a token set with k hash functions."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    """Jaccard‑like similarity based on exact MinHash collisions."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# ----------------------------------------------------------------------
# Parent B – Diffusion schedule
# ----------------------------------------------------------------------
def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """
    Returns a length‑T+1 diffusion schedule α̅ₜ ∈ (0,1].

    * ``cosine`` – cosine‑squared schedule (default).
    * ``linear`` – linearly decreasing schedule derived from a linear β schedule.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        return np.clip(alpha_bars, 1e-9, 1.0)
    elif schedule == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        # cumulative product to obtain α̅ₜ
        alpha_bars = np.cumprod(alphas)
        alpha_bars = np.insert(alpha_bars, 0, 1.0)  # α̅₀ = 1
        return np.clip(alpha_bars, 1e-9, 1.0)
    else:
        raise ValueError(f"Unsupported schedule '{schedule}'")

# ----------------------------------------------------------------------
# Parent A – Normalized LMS utilities
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = wᵀx."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Returns the updated weight vector and the instantaneous error.
    """
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + (mu * error / power) * x
    return next_weights, error

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def _signature_to_feature(sig: list[int], dim: int) -> np.ndarray:
    """
    Convert a MinHash signature to a real‑valued feature vector of length ``dim``.
    The conversion uses a simple modulo‑2 binarisation followed by scaling to [0,1].
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    # Truncate or repeat the signature to match the desired dimension
    repeated = (sig * ((dim // len(sig)) + 1))[:dim]
    binary = np.array([v & 1 for v in repeated], dtype=np.float64)
    return binary  # values are 0.0 or 1.0

def hybrid_compute_scaled_schedule(
    weights: np.ndarray,
    T: int,
    schedule: str = "cosine",
) -> np.ndarray:
    """
    Element‑wise multiplication of the diffusion schedule α̅ₜ by the adaptive
    weight vector **w** (same length).  The result is the effective schedule used
    for prediction.
    """
    base_sched = noise_schedule(T, schedule)
    if len(weights) != len(base_sched):
        raise ValueError("weights and schedule must have the same length")
    return weights * base_sched

def hybrid_predict(
    tokens: list[str],
    weights: np.ndarray,
    T: int,
    schedule: str = "cosine",
) -> float:
    """
    Predict a scalar value for a token set.

    1. Compute a MinHash signature → feature vector x.
    2. Build the scaled diffusion schedule ŝₜ = w ⊙ α̅ₜ.
    3. Return the dot product x·ŝ (interpreted as a linear read‑out).
    """
    sig = signature(tokens, k=len(weights))
    x = _signature_to_feature(sig, dim=len(weights))
    scaled_sched = hybrid_compute_scaled_schedule(weights, T, schedule)
    return float(np.dot(x, scaled_sched))

def hybrid_train(
    init_weights: np.ndarray,
    token_sequences: list[list[str]],
    target_sequence: list[float],
    T: int,
    schedule: str = "cosine",
    mu: float = 0.5,
) -> np.ndarray:
    """
    One‑pass training over a sequence of token sets.

    Parameters
    ----------
    init_weights : np.ndarray
        Initial weight vector (length = T+1).
    token_sequences : list of token lists
        Each element is the token set at a time step.
    target_sequence : list of float
        Desired scalar output for each time step.
    T : int
        Length of the diffusion schedule (must match ``len(init_weights)-1``).
    schedule : str
        Diffusion schedule type (``'cosine'`` or ``'linear'``).
    mu : float
        NLMS step‑size.

    Returns
    -------
    np.ndarray
        The adapted weight vector after processing the whole sequence.
    """
    if len(token_sequences) != len(target_sequence):
        raise ValueError("token_sequences and target_sequence must have equal length")
    if len(init_weights) != T + 1:
        raise ValueError("init_weights length must be T+1 (including α̅₀)")

    w = init_weights.copy()
    for tokens, target in zip(token_sequences, target_sequence):
        # Feature extraction
        sig = signature(tokens, k=len(w))
        x = _signature_to_feature(sig, dim=len(w))

        # Prediction using current weights
        pred = hybrid_predict(tokens, w, T, schedule)

        # NLMS adaptation (treating the scaled schedule as the linear model)
        w, _ = nlms_update(w, x, target, mu=mu)

    return w

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Hyper‑parameters
    T = 10  # schedule length (produces T+1 coefficients)
    schedule_type = "cosine"
    mu = 0.4

    # Initialise weights to ones (neutral scaling)
    init_w = np.ones(T + 1, dtype=np.float64)

    # Dummy token sequence (alternating small vocab)
    token_seq = [
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon"],
        ["zeta", "eta", "theta", "iota"],
        ["kappa"],
        ["lambda", "mu", "nu", "xi", "omicron"],
    ]

    # Synthetic targets – e.g., increasing linearly
    targets = [0.1 * i for i in range(len(token_seq))]

    # Train
    final_weights = hybrid_train(
        init_weights=init_w,
        token_sequences=token_seq,
        target_sequence=targets,
        T=T,
        schedule=schedule_type,
        mu=mu,
    )

    # Show a prediction after training
    test_tokens = ["pi", "rho", "sigma"]
    pred = hybrid_predict(test_tokens, final_weights, T, schedule_type)
    print(f"Final weights: {final_weights}")
    print(f"Prediction for {test_tokens}: {pred:.4f}")