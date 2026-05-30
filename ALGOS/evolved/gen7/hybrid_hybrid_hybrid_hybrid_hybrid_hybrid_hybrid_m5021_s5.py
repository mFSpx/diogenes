# DARWIN HAMMER — match 5021, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s3.py (gen6)
# born: 2026-05-29T23:59:29Z

"""
Hybrid Fusion Module
====================

Parents:
- **Parent A**: `hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s1.py`
  Provides morphological descriptors (sphericity, flatness), Gaussian beam modeling,
  Fisher information score, and MinHash‑based token similarity.

- **Parent B**: `hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s3.py`
  Supplies high‑dimensional binary vectors, random/symbolic generation, binding
  (element‑wise multiplication) and bundling (majority vote) operations.

Mathematical Bridge
-------------------
Both parents employ *information‑theoretic* measures:

* **Fisher score** (Parent A) quantifies the sensitivity of a Gaussian‐beam
  observation with respect to its parameters.
* **Binding/Bundling** (Parent B) creates composite high‑dimensional representations
  that can be interpreted as *probability‑like* feature aggregations.

The fusion uses the Fisher score as a *scalar weight* for binding operations,
thereby turning a high‑dimensional binary vector into a weighted representation
that reflects the underlying morphological certainty.  Conversely, the MinHash
similarity from Parent A is combined with sphericity/flatness‑derived geometric
similarity to produce a unified similarity metric for token sequences.

The three core functions below demonstrate this integration:
1. `hybrid_similarity` – merges geometric and MinHash similarities.
2. `weighted_bind` – binds two vectors and scales the result by a Fisher score.
3. `bundle_morphologies` – converts a collection of `Morphology` objects into
   binary vectors, binds each with its Fisher‑weighted Gaussian beam, and
   bundles the collection into a single composite vector.
"""

import math
import random
import hashlib
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of the three dimensions to the longest side."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length + width) / (2 * height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    """Gaussian intensity scaled by sphericity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float,
                 sphericity: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def _minhash_token(token: str) -> int:
    """Deterministic 128‑bit integer hash of a token."""
    return int(hashlib.md5(token.encode()).hexdigest(), 16)


def minhash_signature(tokens: List[str]) -> int:
    """
    Classic MinHash signature: the minimum hash value over the token set.
    """
    if not tokens:
        raise ValueError("token list cannot be empty")
    return min(_minhash_token(t) for t in tokens)


def minhash_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Approximate Jaccard similarity using a single MinHash signature.
    The probability that the minima coincide equals the Jaccard index.
    """
    sig1 = minhash_signature(tokens1)
    sig2 = minhash_signature(tokens2)
    return 1.0 if sig1 == sig2 else 0.0  # single‑hash estimator


def geometric_similarity(m1: Morphology, m2: Morphology) -> float:
    """
    Combines sphericity and flatness indices of two morphologies into a
    similarity score in [0, 1].
    """
    sph1 = sphericity_index(m1.length, m1.width, m1.height)
    sph2 = sphericity_index(m2.length, m2.width, m2.height)
    flat1 = flatness_index(m1.length, m1.width, m1.height)
    flat2 = flatness_index(m2.length, m2.width, m2.height)

    sph_sim = 1.0 - abs(sph1 - sph2) / max(sph1, sph2, 1e-12)
    flat_sim = 1.0 - abs(flat1 - flat2) / max(flat1, flat2, 1e-12)
    return (sph_sim + flat_sim) / 2.0


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
Vector = List[int]  # binary (+1 / -1) high‑dimensional vector


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    """Generate a random binary vector (+1 / -1)."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """
    Deterministic binary vector derived from a symbolic name.
    The SHA‑256 digest seeds the random generator.
    """
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (binding)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """
    Majority‑vote bundling: for each component take the sign of the sum.
    """
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]


# ----------------------------------------------------------------------
# Hybrid Functions (mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_similarity(tokens1: List[str], tokens2: List[str],
                     morph1: Morphology, morph2: Morphology,
                     alpha: float = 0.5) -> float:
    """
    Unified similarity between two objects.
    - `alpha` balances geometric similarity (Parent A) and MinHash token similarity.
    Returns a value in [0, 1].
    """
    geom_sim = geometric_similarity(morph1, morph2)
    token_sim = minhash_similarity(tokens1, tokens2)
    return alpha * geom_sim + (1.0 - alpha) * token_sim


def weighted_bind(theta: float, center: float, width: float,
                  morph: Morphology, vec_a: Vector, vec_b: Vector) -> Vector:
    """
    Bind two vectors and scale the result by the Fisher information of the
    morphological Gaussian beam.  The scalar weight is inserted by multiplying
    each component (still keeping binary sign) with the sign of the Fisher score.
    """
    # Compute Fisher score (always non‑negative)
    spher = sphericity_index(morph.length, morph.width, morph.height)
    fisher = fisher_score(theta, center, width, spher)

    # Binary sign of Fisher (positive -> keep, zero -> neutral)
    sign = 1 if fisher >= 0 else -1

    bound = bind(vec_a, vec_b)
    # Apply scalar sign while preserving binary nature
    return [sign * x for x in bound]


def bundle_morphologies(morphs: List[Morphology],
                        theta: float, center: float, width: float,
                        dim: int = 10000) -> Vector:
    """
    Convert each `Morphology` into a symbolic vector, bind it with its
    Fisher‑weighted Gaussian beam, and bundle the collection.
    The resulting vector encodes the whole set in a single high‑dimensional
    representation.
    """
    bound_vectors = []
    for i, m in enumerate(morphs):
        # Symbolic identifier based on morphology attributes
        symbol = f"morph_{i}_{m.length:.2f}_{m.width:.2f}_{m.height:.2f}_{m.mass:.2f}"
        base_vec = symbol_vector(symbol, dim)

        # Create a second random vector to bind with
        rand_vec = random_vector(dim, seed=i)

        # Fisher‑weighted bind
        weighted = weighted_bind(theta, center, width, m, base_vec, rand_vec)
        bound_vectors.append(weighted)

    return bundle(bound_vectors)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple token lists
    tokens_a = ["alpha", "beta", "gamma"]
    tokens_b = ["alpha", "delta", "epsilon"]

    # Two morphologies
    morph_a = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)
    morph_b = Morphology(length=2.1, width=1.4, height=1.05, mass=3.1)

    # Hybrid similarity
    sim = hybrid_similarity(tokens_a, tokens_b, morph_a, morph_b, alpha=0.6)
    print(f"Hybrid similarity: {sim:.4f}")

    # Vectors for weighted bind
    vec1 = random_vector(seed=42)
    vec2 = random_vector(seed=99)
    bound = weighted_bind(theta=0.5, center=0.0, width=1.0,
                          morph=morph_a, vec_a=vec1, vec_b=vec2)
    print(f"Weighted bind first 10 components: {bound[:10]}")

    # Bundle a collection of morphologies
    collection = [morph_a, morph_b,
                  Morphology(length=1.8, width=1.2, height=0.9, mass=2.5)]
    bundled = bundle_morphologies(collection, theta=0.3, center=0.0, width=0.8)
    print(f"Bundled vector first 10 components: {bundled[:10]}")

    print("Smoke test completed successfully.")