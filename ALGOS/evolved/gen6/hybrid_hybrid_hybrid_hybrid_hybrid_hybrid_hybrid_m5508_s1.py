# DARWIN HAMMER — match 5508, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1818_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_infotaxis_min_m491_s0.py (gen4)
# born: 2026-05-30T00:02:38Z

"""Hybrid Morphology‑MinHash‑Entropy Fusion
================================================

This module combines the core topologies of the two parent algorithms:

* **Parent A** – Hyperdimensional MinHash Serpentina Self‑Righting Morphology.
  It provides a high‑dimensional morphology vector and a MinHash signature
  for a token set.

* **Parent B** – Pheromone‑Infotaxis MinHash hybrid.
  It supplies a MinHash‑based similarity measure and an entropy‑driven
  expected‑entropy computation for token addition.

**Mathematical bridge**

Both parents operate on *MinHash signatures* that are deterministic 64‑bit
hash arrays.  The bridge is built by:

1. Converting a MinHash signature into a dense hyperdimensional float vector
   (normalising each 64‑bit integer to ``[0, 1]`` and tiling to the required
   dimension).
2. **Binding** this signature vector with the morphology vector using an
   element‑wise multiplication (the classic HDC bind operation).
3. Computing **Shannon entropy** of the bound vector and using the
   **Jaccard‑estimated similarity** between signatures to modulate a
   recovery‑priority score.

The resulting system can rank candidate morphologies (or spans) by a single
scalar that reflects structural similarity, token‑set overlap and
information‑theoretic uncertainty.

"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

MAX64 = (1 << 64) - 1
Vector = np.ndarray


# ----------------------------------------------------------------------
# Shared deterministic 64‑bit hash
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# ----------------------------------------------------------------------
# Parent A – Morphology utilities
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float
    tokens: List[str]


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def morphology_vector(m: Morphology, dim: int = 10_000) -> Vector:
    """
    Hyperdimensional vector representing the physical morphology.
    The vector is seeded from the geometric parameters so that the same
    morphology always yields the same vector.
    """
    seed_bytes = f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    # Uniform distribution in [-1, 1]
    return rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)


# ----------------------------------------------------------------------
# Parent B – Span utilities (used here for entropy‑driven scoring)
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Alias for MinHash signature used by the second parent."""
    return minhash_signature(tokens, k)


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑estimated similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Fusion primitives
def _signature_to_vector(sig: List[int], dim: int) -> Vector:
    """
    Map a MinHash signature to a dense float vector.
    Each 64‑bit integer is normalised to [0, 1] and tiled to fill ``dim``.
    """
    norm = np.array(sig, dtype=np.float64) / MAX64  # shape (k,)
    repeats = int(math.ceil(dim / len(sig)))
    tiled = np.tile(norm, repeats)[:dim]
    # Shift to centred range [-0.5, 0.5] for better binding dynamics
    return (tiled - 0.5).astype(np.float32)


def bind_vectors(a: Vector, b: Vector) -> Vector:
    """
    Hyperdimensional bind operation – element‑wise multiplication.
    The result remains in the same vector space and preserves similarity
    information.
    """
    if a.shape != b.shape:
        raise ValueError("vectors must share the same shape for binding")
    return a * b


def shannon_entropy(vec: Vector, eps: float = 1e-12) -> float:
    """Shannon entropy of the absolute‑value distribution of a vector."""
    abs_vals = np.abs(vec)
    total = abs_vals.sum()
    if total == 0:
        return 0.0
    probs = abs_vals / total + eps
    return -float(np.sum(probs * np.log2(probs)))


# ----------------------------------------------------------------------
# Hybrid operations
def fused_representation(morph: Morphology, dim: int = 10_000) -> Vector:
    """
    Produce the bound hyperdimensional representation of a morphology
    by binding its physical vector with the vectorised MinHash signature
    of its textual tokens.
    """
    morph_vec = morphology_vector(morph, dim)
    sig = minhash_signature(morph.tokens, k=dim // 128 or 128)  # ensure enough hashes
    sig_vec = _signature_to_vector(sig, dim)
    return bind_vectors(morph_vec, sig_vec)


def expected_entropy_for_addition(
    current_tokens: List[str],
    token_to_add: str,
    k: int = 128,
) -> float:
    """
    Expected entropy after adding ``token_to_add`` to ``current_tokens``.
    The expectation is weighted by the similarity between the pre‑ and
    post‑addition MinHash signatures.
    """
    sig_before = signature(current_tokens, k)
    sig_after = signature(current_tokens + [token_to_add], k)
    sim = similarity(sig_before, sig_after)

    vec_before = _signature_to_vector(sig_before, dim=k * 8)
    vec_after = _signature_to_vector(sig_after, dim=k * 8)

    ent_before = shannon_entropy(vec_before)
    ent_after = shannon_entropy(vec_after)

    # Linear interpolation based on similarity (high similarity ⇒ low change)
    return sim * ent_before + (1.0 - sim) * ent_after


def recovery_priority(
    morph: Morphology,
    target: Morphology,
    dim: int = 10_000,
) -> float:
    """
    Unified priority metric combining:
    * similarity of MinHash signatures,
    * entropy of the bound representation,
    * a right‑ing‑time index derived from physical dimensions.
    Higher values indicate a more urgent or promising recovery candidate.
    """
    # 1. Signature similarity
    sim = similarity(minhash_signature(morph.tokens, k=128),
                     minhash_signature(target.tokens, k=128))

    # 2. Entropy of the bound vector (lower entropy ⇒ more certain)
    bound_vec = bind_vectors(
        morphology_vector(morph, dim),
        _signature_to_vector(minhash_signature(morph.tokens, k=128), dim),
    )
    ent = shannon_entropy(bound_vec) + 1e-9  # avoid division by zero

    # 3. Right‑ing‑time index (volume‑to‑mass ratio)
    def righting_time_index(m: Morphology) -> float:
        volume = m.length * m.width * m.height
        return volume / (m.mass + 1e-9)

    rti = righting_time_index(morph)

    # Combine: similarity boosts priority, high entropy penalises it,
    # and the physical right‑ing factor scales the result.
    return sim * (1.0 / ent) * rti


# ----------------------------------------------------------------------
# Smoke test
if __name__ == "__main__":
    # Create two simple morphologies
    morph_a = Morphology(
        length=2.0,
        width=1.5,
        height=0.8,
        mass=3.0,
        tokens=["alpha", "beta", "gamma", "delta"],
    )
    morph_b = Morphology(
        length=2.1,
        width=1.4,
        height=0.9,
        mass=2.9,
        tokens=["alpha", "beta", "epsilon", "zeta"],
    )

    # Compute fused vectors (ensures binding works)
    vec_a = fused_representation(morph_a)
    vec_b = fused_representation(morph_b)

    # Basic sanity checks
    assert vec_a.shape == vec_b.shape == (10_000,)
    assert not np.allclose(vec_a, vec_b)  # likely different

    # Expected entropy when adding a new token to morph_a
    exp_ent = expected_entropy_for_addition(morph_a.tokens, "theta")
    print(f"Expected entropy after addition: {exp_ent:.4f}")

    # Recovery priority score for morph_a w.r.t. morph_b
    priority = recovery_priority(morph_a, morph_b)
    print(f"Recovery priority (A vs B): {priority:.6f}")

    # Ensure similarity is within [0,1]
    sim = similarity(
        minhash_signature(morph_a.tokens),
        minhash_signature(morph_b.tokens),
    )
    print(f"Signature similarity: {sim:.3f}")

    sys.exit(0)