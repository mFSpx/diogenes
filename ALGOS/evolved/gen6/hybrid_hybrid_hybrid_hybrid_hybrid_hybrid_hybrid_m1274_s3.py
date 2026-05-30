# DARWIN HAMMER — match 1274, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m834_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_ternar_m333_s3.py (gen3)
# born: 2026-05-29T23:34:54Z

"""Hybrid Algorithm integrating pheromone‑Fisher information (Parent A) with
weekday‑modulated MinHash sheaf construction (Parent B).

Mathematical Bridge
-------------------
Parent A provides a probability distribution **p** over a set of surface
tokens and a Fisher‑information weight **I(p,θ)** for each token.  
Parent B supplies a deterministic, weekday‑dependent weight vector **w(d)**
and a MinHash‑based sheaf representation **h(p·I)** built from the
product distribution **q = p·I**.

The fusion treats **q** as a measure on the token space.  The weekday
vector **w(d)** linearly modulates **q**, yielding a temporally‑aware
distribution **q̂ = q ⊙ w(d)** (⊙ denotes element‑wise multiplication after
broadcasting).  From **q̂** we generate a MinHash signature, interpreting
each hash function as a section of a sheaf; the minimal hash value over
tokens with non‑zero **q̂** measures local disagreement.  Finally, the
overall hybrid score is a convex combination of the entropy of **q̂**
and the entropy of its MinHash signature, weighted by the VFE factor
**λ** from Parent B.

This module implements the three core operations:
1. construction of the Fisher‑weighted pheromone vector,
2. weekday‑modulated MinHash sheaf generation,
3. hybrid entropy‑based scoring.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
import hashlib

import numpy as np

# ----------------------------------------------------------------------
# Constants (merged from both parents)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
BASE_TAU = 1.0          # baseline liquid time constant (unused in this fusion)
ALPHA = 5.0             # gating steepness (unused in this fusion)
LAMBDA = 0.7            # VFE weighting factor for hybrid entropy
MINHASH_K = 64          # number of hash functions for MinHash
MAX64 = (1 << 64) - 1   # mask for 64‑bit hashing
SEED_BASE = 123456789   # deterministic base seed for all RNGs

# ----------------------------------------------------------------------
# Deterministic RNG (Parent B)
# ----------------------------------------------------------------------
_rng = np.random.default_rng(SEED_BASE)

# ----------------------------------------------------------------------
# Parent A – pheromone & Fisher utilities
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(surface_key: str, limit: int) -> list[float]:
    """Simulate pheromone probabilities for *limit* tokens."""
    # The surface_key is currently unused; kept for API compatibility.
    pheromones = [_rng.random() for _ in range(limit)]
    total = sum(pheromones)
    if total == 0:
        raise ValueError("Pheromone sum cannot be zero.")
    return [p / total for p in pheromones]

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Gaussian Fisher information for a single probability value."""
    intensity = max(math.exp(-0.5 * ((theta - center) / width) ** 2), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a normalized probability vector."""
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("Positive probability mass required.")
    probs = probabilities / total
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))

# ----------------------------------------------------------------------
# Parent B – weekday weight vector
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: tuple[str, ...], dow: int) -> np.ndarray:
    """
    Normalised weight vector w(d) for the given weekday index ``dow``
    (0 = Sun … 6 = Sat).  A sinusoidal pattern with small amplitude avoids
    collapse to a one‑hot vector.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    return raw / raw.sum()

def today_weekday() -> int:
    """Return today's weekday index compatible with ``weekday_weight_vector``."""
    return (date.today().weekday() + 1) % 7  # 0 = Sunday

# ----------------------------------------------------------------------
# MinHash utilities (Parent B, completed)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """
    Deterministic 64‑bit hash based on SHA‑256.
    The result is masked to 64 bits to stay within ``MAX64``.
    """
    h = hashlib.sha256()
    h.update(seed.to_bytes(8, "big"))
    h.update(token.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") & MAX64

def minhash_signature(weighted_vector: np.ndarray, seed: int = SEED_BASE) -> np.ndarray:
    """
    Produce a MinHash signature of length ``MINHASH_K`` from ``weighted_vector``.
    For each hash function *i* we compute the minimum hash value over all
    token indices with non‑zero weight.  The token string is simply the
    integer index, guaranteeing a well‑defined sheaf section.
    """
    indices = np.nonzero(weighted_vector)[0]
    if len(indices) == 0:
        raise ValueError("Weighted vector contains only zeros.")
    signature = np.empty(MINHASH_K, dtype=np.uint64)
    for i in range(MINHASH_K):
        min_val = MAX64
        for idx in indices:
            h = _hash(seed + i, str(idx))
            if h < min_val:
                min_val = h
        signature[i] = min_val
    return signature

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def fisher_weighted_pheromone_vector(
    surface_key: str,
    limit: int,
    center: float,
    width: float
) -> np.ndarray:
    """
    Compute the element‑wise product of pheromone probabilities and their
    Fisher information scores, returning a dense ``np.ndarray`` of shape
    ``(limit,)``.
    """
    pheromone_probs = np.array(calculate_pheromone_probabilities(surface_key, limit))
    fisher_infos = np.vectorize(lambda p: fisher_score(p, center, width))(pheromone_probs)
    return pheromone_probs * fisher_infos

def weekday_modulated_vector(
    base_vector: np.ndarray,
    dow: int
) -> np.ndarray:
    """
    Apply the weekday weight vector to ``base_vector``.  If the number of
    groups differs from ``base_vector`` length, the weight vector is
    broadcast (repeated) to match the length.
    """
    w = weekday_weight_vector(GROUPS, dow)
    # Broadcast weight vector to the length of ``base_vector``.
    repeats = int(np.ceil(len(base_vector) / len(w)))
    w_expanded = np.tile(w, repeats)[: len(base_vector)]
    return base_vector * w_expanded

def hybrid_entropy_score(
    surface_key: str,
    limit: int,
    center: float,
    width: float,
    dow: int | None = None
) -> float:
    """
    Full hybrid scoring pipeline:

    1. Build Fisher‑weighted pheromone vector ``q``.
    2. Modulate ``q`` with weekday weights → ``q̂``.
    3. Generate MinHash signature ``h`` from ``q̂``.
    4. Compute entropies ``H_q̂`` and ``H_h``.
    5. Return convex combination ``λ·H_q̂ + (1‑λ)·H_h``.
    """
    if dow is None:
        dow = today_weekday()
    q = fisher_weighted_pheromone_vector(surface_key, limit, center, width)
    q_mod = weekday_modulated_vector(q, dow)
    # Normalise before entropy to avoid numerical issues.
    H_q = entropy(q_mod)

    signature = minhash_signature(q_mod)
    # Treat the signature as a multiset of hash values; compute its empirical distribution.
    # Convert to frequencies.
    unique, counts = np.unique(signature, return_counts=True)
    freq = counts.astype(np.float64)
    H_h = entropy(freq)

    return LAMBDA * H_q + (1.0 - LAMBDA) * H_h

def disagreement_measure(
    surface_key: str,
    limit: int,
    center: float,
    width: float,
    dow: int | None = None
) -> float:
    """
    Quantify the sheaf‑cohomology‑style disagreement between the
    weekday‑modulated Fisher distribution and its MinHash sheaf.
    The metric is the average absolute difference between the normalized
    weighted vector and the normalized histogram of the MinHash signature.
    """
    if dow is None:
        dow = today_weekday()
    q = fisher_weighted_pheromone_vector(surface_key, limit, center, width)
    q_mod = weekday_modulated_vector(q, dow)
    q_norm = q_mod / q_mod.sum()

    signature = minhash_signature(q_mod)
    unique, counts = np.unique(signature, return_counts=True)
    hist = np.zeros_like(q_norm)
    # Map each unique hash to a position in the vector (modulo length) to obtain a comparable distribution.
    for h, c in zip(unique, counts):
        idx = h % len(hist)
        hist[idx] += c
    hist_norm = hist / hist.sum()

    return float(np.mean(np.abs(q_norm - hist_norm)))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    surface = "example_surface"
    limit = 12
    center = 0.5
    width = 0.15

    score = hybrid_entropy_score(surface, limit, center, width)
    disc = disagreement_measure(surface, limit, center, width)

    print(f"Hybrid entropy score: {score:.6f}")
    print(f"Disagreement measure: {disc:.6f}")