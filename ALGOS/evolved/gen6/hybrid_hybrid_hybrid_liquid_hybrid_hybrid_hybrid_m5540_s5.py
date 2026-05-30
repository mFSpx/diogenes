# DARWIN HAMMER — match 5540, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_liquid_time_c_hybrid_hybrid_korpus_m2174_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2368_s0.py (gen5)
# born: 2026-05-30T00:02:50Z

"""Hybrid Module: hybrid_ltc_sheaf_entropy_ssim

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – `hybrid_hybrid_liquid_time_c_hybrid_hybrid_korpus_m2174_s1.py`  
  Provides a Liquid Time‑Constant (LTC) cell whose hidden state is processed
  through a MinHash signature and a Voronoi partitioning of 2‑D points.

* **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2368_s0.py`  
  Introduces ternary vectors, Shannon entropy of those vectors and a
  Structural Similarity Index (SSIM) that modulates a sheaf‑like energy
  computation.

**Mathematical Bridge**  
The bridge is built on the observation that a MinHash signature can be
binarised/ternarised to obtain a ternary vector. Its Shannon entropy is used
as a temperature‑like scaling factor for the LTC output, while the SSIM
between the current input and a set of reference inputs provides an
additional similarity weight. The final hybrid update therefore combines:


h = sigmoid(W·[x;I] + b)                     # LTC linear‑nonlinear step
τ = entropy( ternary( minhash(h) ) )         # entropy‑derived temperature
σ = mean_{ref} SSIM(I, ref)                  # similarity to references
output = h * τ * σ                          # fused activation


The Voronoi partitioning from Parent A is retained as an auxiliary
geometric view of the hidden state.

The module implements three public functions that demonstrate this hybrid
behaviour and a small smoke test.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import Counter, deque

# ----------------------------------------------------------------------
# Utility primitives (shared by both parents)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        __import__("hashlib").blake2b(data, digest_size=8).digest(), "big"
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Parent A – MinHash & Voronoi utilities
# ----------------------------------------------------------------------
def minhash_signature(arr: list[str], k: int = 64) -> list[int]:
    """Compute a simple MinHash signature of length k."""
    signature = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    for s in arr:
        h = _hash(0, s) % (2 ** 31 - 1)
        idx = h % k
        signature[idx] = min(signature[idx], h)
    return signature.tolist()


def ternary_from_signature(sig: list[int]) -> np.ndarray:
    """Map a MinHash signature to a ternary vector (-1, 0, 1)."""
    arr = np.array(sig, dtype=np.int64)
    q1 = np.percentile(arr, 33)
    q2 = np.percentile(arr, 66)
    tern = np.where(arr <= q1, -1, np.where(arr >= q2, 1, 0))
    return tern


def shannon_entropy(ternary_vec: np.ndarray) -> float:
    """Shannon entropy of a ternary vector."""
    counter = Counter(ternary_vec.tolist())
    total = sum(counter.values())
    entropy = 0.0
    for cnt in counter.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def ssim_index(x: np.ndarray, y: np.ndarray, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
    """A lightweight SSIM implementation for 1‑D signals."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0


def voronoi_assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    """Assign each point to the nearest seed (Voronoi region)."""
    if not seeds:
        raise ValueError("At least one seed required")
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        dists = [math.hypot(p[0] - s[0], p[1] - s[1]) for s in seeds]
        regions[int(np.argmin(dists))].append(p)
    return regions


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_ltc_hidden(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Standard LTC hidden state computation:
        h = sigmoid(W·[x;I] + b)
    """
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)


def entropy_modulation_factor(hidden: np.ndarray, k: int = 64) -> float:
    """
    Derive a scaling factor from the entropy of a ternary vector obtained
    from a MinHash of the hidden state.
    """
    # Convert hidden state to string tokens for MinHash
    tokens = [f"{v:.6f}" for v in hidden]
    sig = minhash_signature(tokens, k=k)
    tern = ternary_from_signature(sig)
    ent = shannon_entropy(tern)
    # Normalise entropy to (0,1] by dividing by log2(3) (max entropy for ternary)
    max_ent = math.log2(3)
    return ent / max_ent if max_ent != 0 else 0.0


def hybrid_ltc_update(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    reference_inputs: list[np.ndarray],
    k: int = 64,
    num_seeds: int = 5,
) -> dict:
    """
    Perform a full hybrid update:

    1. Compute LTC hidden state h.
    2. Compute entropy‑based temperature τ from h.
    3. Compute SSIM similarity σ between I and each reference, average them.
    4. Produce final output = h * τ * σ.
    5. Additionally, map h to 2‑D points (first two dimensions) and return a
       Voronoi partition of those points using randomly generated seeds.

    Returns a dictionary with keys:
        'hidden'   – raw LTC hidden state (np.ndarray)
        'output'   – entropy‑ and SSIM‑modulated activation (np.ndarray)
        'tau'      – entropy scaling factor (float)
        'sigma'    – average SSIM similarity (float)
        'voronoi'  – Voronoi region mapping (dict[int, list[tuple[float,float]]])
    """
    # 1. LTC hidden state
    h = compute_ltc_hidden(x, I, W, b)

    # 2. Entropy scaling
    tau = entropy_modulation_factor(h, k=k)

    # 3. SSIM similarity to references
    if reference_inputs:
        sims = [ssim_index(I, ref) for ref in reference_inputs]
        sigma = float(np.mean(sims))
    else:
        sigma = 1.0

    # 4. Modulated output
    output = h * tau * sigma

    # 5. Voronoi partition (use first two dims of h as points)
    if h.size >= 2:
        points = [(float(h[i]), float(h[i + 1])) for i in range(0, h.size - 1, 2)]
    else:
        points = [(float(h[0]), float(h[0]))]

    # Random seeds in the same 2‑D space
    rng = np.random.default_rng()
    seeds = [(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))) for _ in range(num_seeds)]
    vor = voronoi_assign(points, seeds)

    return {
        "hidden": h,
        "output": output,
        "tau": tau,
        "sigma": sigma,
        "voronoi": vor,
    }


def generate_random_ltc_parameters(dim_x: int, dim_I: int, hidden_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Utility to create random LTC parameters:
        W : (hidden_dim, dim_x + dim_I)
        b : (hidden_dim,)
        x : (dim_x,)
        I : (dim_I,)
    """
    rng = np.random.default_rng()
    W = rng.normal(loc=0.0, scale=1.0, size=(hidden_dim, dim_x + dim_I))
    b = rng.normal(loc=0.0, scale=0.5, size=hidden_dim)
    x = rng.normal(loc=0.0, scale=1.0, size=dim_x)
    I = rng.normal(loc=0.0, scale=1.0, size=dim_I)
    return x, I, W, b


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensions
    dim_x, dim_I, hidden = 4, 3, 6

    # Random LTC parameters and inputs
    x, I, W, b = generate_random_ltc_parameters(dim_x, dim_I, hidden)

    # Create a few reference inputs for SSIM (same shape as I)
    ref_inputs = [np.random.normal(size=dim_I) for _ in range(3)]

    # Run hybrid update
    result = hybrid_ltc_update(
        x=x,
        I=I,
        W=W,
        b=b,
        reference_inputs=ref_inputs,
        k=64,
        num_seeds=4,
    )

    # Simple sanity prints (no external dependencies)
    print("Hidden state:", result["hidden"])
    print("Output (modulated):", result["output"])
    print("Entropy factor τ:", result["tau"])
    print("SSIM similarity σ:", result["sigma"])
    print("Voronoi regions (counts):", {k: len(v) for k, v in result["voronoi"].items()})