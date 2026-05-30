# DARWIN HAMMER — match 1404, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_caputo_serpentina_self_righ_m444_s0.py (gen4)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_voronoi_parti_m453_s0.py (gen3)
# born: 2026-05-29T23:36:03Z

"""Hybrid Caputo‑MinHash (HCM) algorithm.

Parents:
- hybrid_hybrid_hybrid_caputo_serpentina_self_righ_m444_s0.py
- hybrid_hybrid_infotaxis_min_hybrid_voronoi_parti_m453_s0.py

Mathematical bridge:
The Caputo fractional‑derivative kernel provides a set of long‑range memory weights
w_i = Δt_i^{‑α}.  By interpreting the normalized weight vector as a phase
φ = 2π·∑w_i we obtain a rotation angle that can be used as a GA‑rotor
(in a 2‑D sub‑space) to rotate any vector.  MinHash signatures are integer
vectors that encode set similarity.  Rotating the first two components of a
signature yields a continuous embedding that can be compared with cosine
similarity, while the original integer signature still supplies an
approximate Jaccard estimate.  The hybrid similarity combines both measures,
thereby fusing fractional‑memory‑driven geometry with set‑based similarity."""

import math
import random
import sys
import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

MAX64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# Parent A – fractional‑derivative utilities
# ----------------------------------------------------------------------
def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of Gamma(z) for z > 0."""
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
    Compute normalized Caputo kernel weights for a monotonic time vector.

    w_i ∝ (Δt_i)^{‑α}  where Δt_i = t_i – t_{i‑1}.
    The returned array has length len(times)‑1 and sums to 1.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1) for a proper Caputo kernel")
    if len(times) < 2:
        raise ValueError("need at least two time stamps")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("time vector must be strictly increasing")
    raw = dt ** (-alpha)
    return raw / raw.sum()


def rotor_angle_from_weights(weights: np.ndarray) -> float:
    """
    Map a weight vector to a rotation angle φ ∈ [0, 2π).

    The mapping uses the cumulative sum as a phase accumulator.
    """
    phase = weights.sum()  # equals 1 after normalization, but we keep generality
    return (2 * math.pi * phase) % (2 * math.pi)


# ----------------------------------------------------------------------
# Parent B – MinHash utilities
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def rotate_signature(sig: List[int], angle: float) -> np.ndarray:
    """
    Rotate the first two components of a signature vector by *angle*.

    The signature is interpreted as a real vector; the rotation is a
    standard 2‑D orthogonal transformation applied to indices 0 and 1.
    """
    vec = np.array(sig, dtype=float)
    if vec.size < 2:
        return vec  # nothing to rotate
    c, s = math.cos(angle), math.sin(angle)
    x, y = vec[0], vec[1]
    vec[0] = c * x - s * y
    vec[1] = s * x + c * y
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity between two real vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def hybrid_similarity(
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    times: List[float],
    alpha: float = 0.5,
    k: int = 128,
    beta: float = 0.5,
) -> float:
    """
    Compute a fused similarity measure.

    Steps
    -----
    1. Produce MinHash signatures for both token sets.
    2. Derive Caputo fractional weights from the supplied time stamps.
    3. Convert the weight vector into a rotation angle (GA‑rotor).
    4. Rotate the first two components of each signature.
    5. Compute cosine similarity of the rotated real vectors.
    6. Compute Jaccard similarity of the original integer signatures.
    7. Return a convex combination:
           β * cosine + (1‑β) * Jaccard
    """
    # 1. MinHash signatures
    sig_a = signature(tokens_a, k=k)
    sig_b = signature(tokens_b, k=k)

    # 2. Fractional weights
    w = caputo_weights(times, alpha)

    # 3. Rotation angle
    theta = rotor_angle_from_weights(w)

    # 4. Rotate signatures
    rot_a = rotate_signature(sig_a, theta)
    rot_b = rotate_signature(sig_b, theta)

    # 5. Cosine similarity (continuous embedding)
    cos_sim = cosine_similarity(rot_a, rot_b)

    # 6. Jaccard similarity (discrete MinHash)
    jac_sim = jaccard_similarity(sig_a, sig_b)

    # 7. Blend
    return beta * cos_sim + (1.0 - beta) * jac_sim


def voronoi_weighted_distance(
    points: List[Tuple[float, float]],
    query: Tuple[float, float],
    reliability: List[float],
) -> float:
    """
    Compute a reliability‑weighted Euclidean distance from *query* to the
    nearest point in a Voronoi‑like sense.

    The distance is d = min_i (‖p_i‑query‖ / r_i) where r_i is the reliability
    score (higher reliability → smaller penalty).
    """
    if len(points) != len(reliability):
        raise ValueError("points and reliability must have the same length")
    min_val = float("inf")
    qx, qy = query
    for (px, py), r in zip(points, reliability):
        if r <= 0:
            continue  # ignore non‑positive reliability
        d = math.hypot(px - qx, py - qy) / r
        if d < min_val:
            min_val = d
    return min_val if min_val != float("inf") else float("nan")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example token sets
    tokens1 = ["alpha", "beta", "gamma", "delta", "epsilon"]
    tokens2 = ["beta", "delta", "zeta", "eta", "theta"]

    # Time stamps for the fractional kernel (e.g., measurement times)
    times = [0.0, 1.0, 2.5, 4.0, 6.0]  # monotonic increasing

    # Compute hybrid similarity
    sim = hybrid_similarity(
        tokens_a=tokens1,
        tokens_b=tokens2,
        times=times,
        alpha=0.4,
        k=64,
        beta=0.6,
    )
    print(f"Hybrid similarity (combined): {sim:.4f}")

    # Demonstrate Voronoi‑weighted distance
    pts = [(0.0, 0.0), (2.0, 2.0), (5.0, 1.0)]
    rel = [0.9, 0.5, 0.8]  # reliability scores
    query_pt = (1.0, 1.5)
    dist = voronoi_weighted_distance(pts, query_pt, rel)
    print(f"Reliability‑weighted distance to query point: {dist:.4f}")