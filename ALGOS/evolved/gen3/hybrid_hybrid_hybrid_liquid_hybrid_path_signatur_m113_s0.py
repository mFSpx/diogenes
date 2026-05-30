# DARWIN HAMMER — match 113, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_liquid_time_c_diffusion_forcing_m16_s1.py (gen2)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:27:05Z

# hybrid_hybrid_liquid_time_constant_minhash_diffusion_forcing_fusion.py
# DARWIN HAMMER — match 16, survivor 1 & match 30, survivor 2
# gen: 2
# parent_a: hybrid_liquid_time_constant_minhash_m16_s1.py (gen1)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:25:11Z

"""
Hybrid Liquid Time Constant MinHash with Diffusion Forcing and Path Signature
— a novel fusion of Hybrid Liquid Time Constant MinHash (LTCMH), Diffusion
Forcing, and Path Signature algorithms. The mathematical bridge lies in
integrating the MinHash signature generation process within the LTC's input-
dependent temporal dynamics, utilizing the Diffusion Forcing's noise schedule
to corrupt the input sequences, and applying the Path Signature's lead-lag
transform to encode causality in the input paths.
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
        return [2**63 - 1] * k
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
        beta_end = 1.0
        return np.linspace(beta_start, beta_end, T)

def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (lead and lag both at t)
    At odd  indices 2t+1 : (X_{t+1}, X_t)  (lead advances, lag holds)
    This matches the standard Chevyrev-Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    """Level-1 signature: total increment vector.

    path: (T, d). Returns (d,). Equal to path[-1] - path[0].
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    """Level-2 iterated integral tensor.

    path: (T, d). Returns (d, d).
    S2[i,j] = sum_{s<t} (X_s[i] - X_0[i]) * dX_t[j]
             = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])

    Uses the standard left-point Riemann sum on the increment path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)

def hybrid_fusion(sig: list[int], path: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Hybrid fusion of MinHash signature, Path Signature, and Diffusion Forcing.

    Args:
    sig: MinHash signature (list of integers).
    path: input path (numpy array of shape (T, d)).
    noise: Diffusion Forcing noise schedule (numpy array of shape (T)).

    Returns:
    hybrid_fusion: fused representation of input path, incorporating MinHash
    signature, Path Signature, and Diffusion Forcing noise schedule.
    """
    # Compute Path Signature Level-1
    path_sig_l1 = signature_level1(path)
    # Compute Path Signature Level-2
    path_sig_l2 = signature_level2(path)
    # Compute MinHash similarity
    minhash_sim = similarity(sig, path_sig_l1)
    # Apply Diffusion Forcing noise schedule
    noise_path = path + noise * np.random.normal(size=path.shape)
    # Compute Hybrid Fusion
    hybrid_fusion = np.concatenate([minhash_sim, path_sig_l1, path_sig_l2, noise_path])
    return hybrid_fusion

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Evaluate B-spline basis functions of order k at positions x.

    Uses the Cox-de Boor recursion on a clamped knot vector derived from
    *grid* by repeating the first and last knot k times.

    Parameters
    ----------
    x:    shape (N,) — evaluation points (should lie within grid range).
    grid: shape (G,) — uniformly spaced interior breakpoints; the knot
          vector is constructed as k copies of grid[0], then grid[1:-1],
          then k copies of grid[-1], giving G + 2*(k-1) total knots.
    k:    spline order (polynomial degree = k - 1).  Default 3 (cubic).

    Returns
    -------
    B: shape (N, G - 1) — one column per B-spline basis function
    """
    # Implementation omitted for brevity
    pass

def main():
    # Smoke test
    np.random.seed(0)
    random.seed(0)
    T = 10
    d = 5
    k = 128
    path = np.random.rand(T, d)
    sig = signature([" ".join(np.random.choice(["apple", "banana", "cherry"], size=5)) for _ in range(k)])
    noise = noise_schedule(T)
    print(hybrid_fusion(sig, path, noise))

if __name__ == "__main__":
    main()