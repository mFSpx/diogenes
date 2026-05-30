# DARWIN HAMMER — match 1388, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m928_s1.py (gen5)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s2.py (gen1)
# born: 2026-05-29T23:35:49Z

"""Hybrid module integrating:
- Parent A: binary high‑dimensional vector algebra (bind, bundle, similarity, Fisher weighting).
- Parent B: text stylometry feature extraction and bilinear model compatibility (vᵀ P m).

Mathematical bridge:
Parent A yields a binary vector **b** ∈ {‑1, 1}ᴰ.  
Parent B supplies a real‑valued feature vector **v** ∈ ℝᴺ (N≥2) from text.
We first map **b** to ℝᴰ by the identity embedding (‑1→‑1.0, 1→1.0) and optionally truncate or pad to match the first two dimensions of **v**.
A projection matrix **P** ∈ ℝᴺˣ² selects the first two components of **v** (mean stylometry and word‑ratio) and maps them into the model space.  
The hybrid score is then

    s = (b̂ ⊙ w)ᵀ (v̂)  where  b̂ = bind‑bundle binary vector,
                                   w = fisher_score(θ) applied component‑wise,
                                   v̂ = Pᵀ (v)  (a 2‑D model‑space vector).

Thus the binary combinatorial structure (bind/bundle) is weighted by a Fisher‑derived Gaussian kernel and finally evaluated through the bilinear form of Parent B. The three core functions below expose this pipeline.
"""

import math
import random
import hashlib
import datetime as dt
import pathlib
import sys
from typing import List, Tuple

import numpy as np

Vector = List[int]          # binary ±1 vector from Parent A
RealVector = np.ndarray     # real‑valued vector from Parent B


# ----------------------------------------------------------------------
# Parent A utilities (binary high‑dimensional algebra)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic binary vector derived from a symbol."""
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Component‑wise multiplication (binding)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """Superposition (majority vote)."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vectors)]
    return [1 if s >= 0 else -1 for s in summed]


def similarity(a: Vector, b: Vector) -> float:
    """Cosine‑like similarity for binary vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Gaussian‑shaped Fisher weighting."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z) + eps


# ----------------------------------------------------------------------
# Parent B utilities (stylometry → model compatibility)
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    """Very simple tokeniser."""
    import re
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def stylometry_features(text: str) -> Tuple[float, float]:
    """
    Returns a 2‑D feature tuple:
    - mean word length,
    - word‑to‑character ratio (words / characters).
    """
    w = words(text)
    if not w:
        return 0.0, 0.0
    mean_len = sum(len(tok) for tok in w) / len(w)
    chars = len(text.replace(" ", ""))
    ratio = len(w) / chars if chars > 0 else 0.0
    return mean_len, ratio


def model_compatibility(feature_vec: RealVector, model_vec: RealVector, P: RealVector) -> float:
    """
    Bilinear compatibility s = vᵀ P m.
    feature_vec shape (N,), model_vec shape (2,), P shape (N,2).
    """
    return float(feature_vec @ (P @ model_vec))


# ----------------------------------------------------------------------
# Hybrid operations (mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_bind_bundle(symbols: List[str], dim: int = 1024) -> Vector:
    """
    Produce a single binary vector from a list of symbols by
    binding each symbol vector with a global seed vector and then bundling.
    """
    seed_vec = random_vector(dim, seed=0xDEADBEEF)
    bound = [bind(seed_vec, symbol_vector(sym, dim)) for sym in symbols]
    return bundle(bound)


def fisher_weighted_similarity(bin_vec: Vector, real_vec: RealVector) -> float:
    """
    Compute similarity between a binary vector and a real vector after
    Fisher weighting.  The real vector is first normalised to the range
    [-1, 1] to match the binary scale.
    """
    # Normalise real vector component‑wise to [-1,1]
    max_abs = np.max(np.abs(real_vec)) if np.max(np.abs(real_vec)) != 0 else 1.0
    norm_real = (real_vec / max_abs).astype(float)

    # Apply Fisher weighting per component (using the binary value as theta)
    weighted = [
        fisher_score(theta=float(b), center=0.0, width=1.0) * r
        for b, r in zip(bin_vec[: len(norm_real)], norm_real)
    ]
    # Cosine‑like dot product
    return float(np.dot(weighted, norm_real) / len(weighted))


def hybrid_route(text: str, symbols: List[str], model_vec: RealVector) -> float:
    """
    Full hybrid pipeline:
    1. Extract 2‑D stylometry feature vector v.
    2. Build binary vector b from symbols (bind+bundle).
    3. Weight b against v via Fisher‑weighted similarity → w.
    4. Form projection matrix P that selects the first two dimensions of v.
    5. Compute final compatibility s = w * (vᵀ P m).
    """
    # Step 1: stylometry (real 2‑D vector)
    mean_len, ratio = stylometry_features(text)
    v = np.array([mean_len, ratio], dtype=float)

    # Step 2: binary vector from symbols
    b = hybrid_bind_bundle(symbols, dim=1024)

    # Step 3: Fisher‑weighted similarity (scalar)
    w = fisher_weighted_similarity(b, v)

    # Step 4: projection matrix P (2×2 identity for the two features)
    P = np.eye(2, dtype=float)

    # Step 5: bilinear model compatibility
    s = model_compatibility(v, model_vec, P)

    # Combine the two scores multiplicatively (any monotone combination works)
    return w * s


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal deterministic test
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_symbols = ["α", "β", "γ", "δ"]
    # Model resource vector: [RAM (scaled), tier]
    model_resource = np.array([0.8, 2.0], dtype=float)  # e.g., 80 % RAM, tier 2

    score = hybrid_route(sample_text, sample_symbols, model_resource)
    print(f"Hybrid compatibility score: {score:.6f}")
    # Ensure no exceptions and a finite number
    assert math.isfinite(score)