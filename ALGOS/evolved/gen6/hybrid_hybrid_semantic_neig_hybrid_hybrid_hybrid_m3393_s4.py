# DARWIN HAMMER — match 3393, survivor 4
# gen: 6
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s5.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s5.py (gen5)
# born: 2026-05-29T23:49:57Z

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Morphology & Recovery Priority
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * max(height, 1e-8))


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority p ∈ [0,1] derived from righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Fractional‑Derivative Utilities
# ----------------------------------------------------------------------
def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of Γ(z) for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    g = 7
    z += g + 0.5
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    a = 0.99999999999980993
    for i, c in enumerate(p[1:], 1):
        a += c / (z - i)
    t = z + g + 0.5
    return math.sqrt(2 * math.pi) * t ** (z - 0.5) * math.exp(-t) * a


def caputo_weights(times: List[float], alpha: float) -> np.ndarray:
    """
    Normalized Caputo kernel weights w_i ∝ (Δt_i)^{‑α}.
    Returns an array of length len(times)‑1 that sums to 1.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if len(times) < 2:
        raise ValueError("need at least two timestamps")
    dt = np.diff(np.asarray(times, dtype=float))
    if np.any(dt <= 0):
        raise ValueError("times must be strictly increasing")
    raw = dt ** (-alpha)
    return raw / raw.sum()


def caputo_phase(times: List[float], alpha: float) -> float:
    """
    Phase Φ = 2π·∑ w_i where w_i are Caputo weights.
    """
    w = caputo_weights(times, alpha)
    return 2.0 * math.pi * w.sum()


# ----------------------------------------------------------------------
# MinHash utilities
# ----------------------------------------------------------------------
def simple_minhash(tokens: Iterable[str], n_hashes: int = 64) -> np.ndarray:
    """
    Produce a deterministic integer MinHash signature of length `n_hashes`.
    For each hash function we use a different random seed mixed with the token.
    """
    rng = np.random.default_rng(12345)  # fixed seed for reproducibility
    seeds = rng.integers(0, 2**32, size=n_hashes, dtype=np.uint32)

    signature = np.full(n_hashes, np.uint64(2**64 - 1), dtype=np.uint64)
    for token in tokens:
        token_bytes = token.encode('utf-8')
        for i, seed in enumerate(seeds):
            h = int.from_bytes(
                (seed.to_bytes(4, 'little') + token_bytes),
                'little', signed=False
            )
            h ^= (h >> 33)
            h *= 0xff51afd7ed558ccd
            h ^= (h >> 33)
            h *= 0xc4ceb9fe1a85ec53
            h ^= (h >> 33)
            signature[i] = min(signature[i], h & ((1 << 64) - 1))
    return signature


def rotate_vector(v: np.ndarray, phi: float) -> np.ndarray:
    """
    Rotate the first two components of `v` by angle `phi` (radians).
    The rest of the vector is left unchanged.
    """
    if v.shape[0] < 2:
        raise ValueError("vector must have at least two dimensions")
    c, s = math.cos(phi), math.sin(phi)
    x, y = v[0], v[1]
    v_rot = v.astype(float).copy()
    v_rot[0] = c * x - s * y
    v_rot[1] = s * x + c * y
    return v_rot


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity in [-1, 1]."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_affinity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    morph_b: Morphology,
    times: List[float],
    alpha: float = 0.5,
) -> float:
    """
    Compute the hybrid affinity between two items.

    Steps:
    1. Compute recovery priority `p` from the morphology of the *candidate* (b).
    2. Derive Caputo phase `Φ` from the provided timestamps and `alpha`.
    3. Rotate `vec_b` by `Φ` → `vec_b_rot`.
    4. Combine plain cosine similarity and rotated cosine similarity:
           h = p * cos(vec_a, vec_b_rot) + (1‑p) * cos(vec_a, vec_b)

    The result lies in [-1, 1] and reflects both morphological priority and
    fractional‑memory‑driven geometry.
    """
    p = recovery_priority(morph_b)
    phi = caputo_phase(times, alpha)
    vec_b_rot = rotate_vector(vec_b, phi)

    cos_plain = cosine_similarity(vec_a, vec_b)
    cos_rot = cosine_similarity(vec_a, vec_b_rot)

    return p * cos_rot + (1.0 - p) * cos_plain


def hybrid_signature_similarity(
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    morph_b: Morphology,
    times: List[float],
    alpha: float = 0.5,
    n_hashes: int = 64,
) -> Tuple[float, float]:
    """
    Returns a tuple (h_affinity, jaccard_estimate).

    - `h_affinity` is the hybrid affinity computed on the rotated MinHash vectors.
    - `jaccard_estimate` is the classic MinHash Jaccard estimator (intersection /
      union of signatures).

    This demonstrates the fusion of set‑based similarity with the continuous
    hybrid geometry.
    """
    sig_a = simple_minhash(tokens_a, n_hashes).astype(float)
    sig_b = simple_minhash(tokens_b, n_hashes).astype(float)

    jaccard_estimate = 1.0 - np.mean(np.maximum(sig_a, sig_b) - np.minimum(sig_a, sig_b), dtype=float)
    h_affinity = hybrid_affinity(sig_a, sig_b, morph_b, times, alpha)

    return h_affinity, jaccard_estimate


def main():
    tokens_a = ["apple", "banana", "orange"]
    tokens_b = ["apple", "banana", "pear"]
    morph_b = Morphology(10.0, 5.0, 2.0, 1.0)
    times = [1.0, 2.0, 3.0]
    alpha = 0.5

    h_affinity, jaccard_estimate = hybrid_signature_similarity(tokens_a, tokens_b, morph_b, times, alpha)
    print(f"Hybrid Affinity: {h_affinity:.4f}, Jaccard Estimate: {jaccard_estimate:.4f}")


if __name__ == "__main__":
    main()