# DARWIN HAMMER — match 113, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_liquid_time_c_diffusion_forcing_m16_s1.py (gen2)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:27:05Z

"""Hybrid MinHash‑Path‑Signature with Diffusion Forcing.

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Hybrid Liquid‑Time‑Constant MinHash with Diffusion Forcing.
  It provides a MinHash based similarity estimator, a shingling routine and a
  diffusion‑forcing noise schedule (cosine / linear).

* **Parent B** – Path Signature (lead‑lag transform, level‑1 & level‑2 signatures).
  It computes iterated integrals of a multivariate time‑series.

**Mathematical bridge**

The similarity `sim` between two tokenised texts (Parent A) is used as a
scalar that conditions the diffusion‑forcing noise schedule applied to a
continuous‑time path before the path signatures are evaluated (Parent B).
Thus the MinHash similarity modulates the stochastic perturbation of the
path, and the resulting noisy path is fed into the level‑1 and level‑2
signatures. The three public functions demonstrate this hybrid pipeline."""

import numpy as np
import hashlib
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Parent A – MinHash, shingles, similarity, diffusion schedule
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: list[str], k: int = 128) -> list[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def shingles(text: str, width: int = 5) -> set[str]:
    """Extract width‑wise word shingles from a string."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """
    Diffusion‑forcing schedule ᾱₜ ∈ (0,1] for t=0…T.
    Cosine schedule follows the DDPM formulation; linear is a simple
    decreasing schedule.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    steps = np.arange(T + 1, dtype=np.float64)

    if schedule == "cosine":
        s = 0.008
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, 1e-9, 1.0)
        return alpha_bars
    elif schedule == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, T + 1, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        alpha_bars = np.clip(alpha_bars, 1e-9, 1.0)
        return alpha_bars
    else:
        raise ValueError(f"Unsupported schedule '{schedule}'")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Parent B – Lead‑lag transform and path signatures
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels.
    Input shape (T, d) → output shape (2T‑1, 2d).
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be 2‑dimensional (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Level‑1 signature: total increment."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """
    Level‑2 iterated integral tensor S₂[i,j] = Σ_{t} (X_{t‑1}[i]‑X₀[i])·ΔX_t[j].
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_similarity_from_texts(text_a: str, text_b: str, k: int = 128, width: int = 5) -> float:
    """
    Compute MinHash similarity between two texts using shingling.
    """
    sh_a = list(shingles(text_a, width))
    sh_b = list(shingles(text_b, width))
    sig_a = minhash_signature(sh_a, k)
    sig_b = minhash_signature(sh_b, k)
    return similarity(sig_a, sig_b)


def diffuse_path(path: np.ndarray, alpha_bars: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """
    Apply diffusion forcing to a path.

    For each step t we add Gaussian noise with variance (1‑ᾱₜ)·amplitude².
    The noise is added to the *increments* so that the overall trajectory
    remains anchored at the original start point.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    if alpha_bars.shape[0] != T:
        raise ValueError("alpha_bars length must match number of timesteps")
    increments = np.diff(path, axis=0)                     # (T‑1, d)
    noisy_increments = np.empty_like(increments)

    for t in range(T - 1):
        var = max(0.0, 1.0 - alpha_bars[t + 1]) * (amplitude ** 2)
        noise = np.random.normal(loc=0.0, scale=math.sqrt(var), size=d)
        noisy_increments[t] = increments[t] + noise

    # Reconstruct the noisy path (keep the original start point)
    noisy_path = np.empty_like(path)
    noisy_path[0] = path[0]
    noisy_path[1:] = path[0] + np.cumsum(noisy_increments, axis=0)
    return noisy_path


def hybrid_signature(
    text_a: str,
    text_b: str,
    raw_path: np.ndarray,
    k: int = 128,
    width: int = 5,
    schedule: str = "cosine",
) -> dict:
    """
    End‑to‑end hybrid pipeline:

    1. Compute MinHash similarity `sim` between `text_a` and `text_b`.
    2. Build a diffusion schedule ᾱₜ for the length of `raw_path`.
    3. Modulate the schedule with `sim` (higher similarity → less noise).
    4. Apply diffusion forcing to `raw_path`.
    5. Perform lead‑lag transform and compute level‑1 & level‑2 signatures.

    Returns a dictionary containing the similarity, the noisy path,
    and both signatures.
    """
    # 1. similarity
    sim = compute_similarity_from_texts(text_a, text_b, k=k, width=width)

    # 2. schedule (length = len(raw_path) because we need T steps)
    T = raw_path.shape[0] - 1  # number of increments
    if T <= 0:
        raise ValueError("raw_path must contain at least two points")
    alpha_bars = noise_schedule(T, schedule=schedule)

    # 3. modulate: map similarity ∈ [0,1] to an amplitude factor.
    #    We let amplitude = 1 - sim so that identical texts produce almost
    #    noiseless paths.
    amplitude = 1.0 - sim

    # 4. diffusion
    noisy_path = diffuse_path(raw_path, alpha_bars, amplitude=amplitude)

    # 5. lead‑lag and signatures
    ll_path = lead_lag_transform(noisy_path)
    lvl1 = signature_level1(ll_path)
    lvl2 = signature_level2(ll_path)

    return {
        "similarity": sim,
        "amplitude": amplitude,
        "alpha_bars": alpha_bars,
        "noisy_path": noisy_path,
        "lead_lag_path": ll_path,
        "level1_signature": lvl1,
        "level2_signature": lvl2,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    txt1 = "the quick brown fox jumps over the lazy dog"
    txt2 = "the quick brown fox leaps over a lazy dog"

    # Generate a synthetic 2‑D random walk of length 20
    steps = np.random.normal(size=(20, 2))
    path = np.cumsum(steps, axis=0)

    result = hybrid_signature(txt1, txt2, path, schedule="cosine")

    print("MinHash similarity :", result["similarity"])
    print("Amplitude (noise scale) :", result["amplitude"])
    print("Level‑1 signature :", result["level1_signature"])
    print("Level‑2 signature shape :", result["level2_signature"].shape)
    # Ensure no exception was raised
    sys.exit(0)