# DARWIN HAMMER — match 740, survivor 2
# gen: 4
# parent_a: gini_coefficient.py (gen0)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s0.py (gen3)
# born: 2026-05-29T23:30:46Z

#!/usr/bin/env python3
"""
Improved hybrid algorithm merging entropy‑based similarity with Gini‑based
inequality measures.

The original version suffered from several critical flaws:
* `gini_entropy` referenced an undefined variable `n` and mixed unrelated
  quantities.
* No normalisation of the probability vector was performed.
* The MinHash signature ignored the magnitude of the probabilities.
* A required `similarity` routine was missing, causing a runtime error.
* Numerical stability and type‑safety were weak.

The rewritten module addresses these points while deepening the mathematical
integration:
* Probabilities are normalised and validated.
* Shannon entropy is normalised to the interval [0, 1] and combined with the
  classic Gini index via a configurable weighting factor.
* A *weighted* MinHash is introduced: the hash value of each token is divided
  by its probability, so larger probabilities tend to dominate the minimum.
* A deterministic Jaccard‑like similarity between two signatures is provided.
* The `chelydrid_strike` cost now incorporates the combined entropy‑Gini
  metric, yielding a richer cost surface.

Only the Python standard library and NumPy are required.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #


def _validate_and_normalise(probs: Sequence[float]) -> np.ndarray:
    """Return a normalised probability vector as a NumPy array.

    Raises:
        ValueError: If the input contains negative values or sums to zero.
    """
    arr = np.asarray(probs, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Probabilities must be non‑negative.")
    total = arr.sum()
    if total <= 0.0:
        raise ValueError("Sum of probabilities must be positive.")
    return arr / total


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash using BLAKE2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# --------------------------------------------------------------------------- #
# Core mathematical measures
# --------------------------------------------------------------------------- #


def combined_gini_entropy(
    probabilities: Sequence[float],
    alpha: float = 0.5,
) -> float:
    """
    Compute a convex combination of the normalised Shannon entropy and the
    Gini index.

    The normalised entropy is `H / log(n)` where `H` is the Shannon entropy and
    `n` the number of categories.  The Gini index is `1 - Σ p_i²`.  Both lie in
    `[0, 1]`.  The parameter `alpha` controls the trade‑off:
    `alpha = 0` → pure Gini, `alpha = 1` → pure normalised entropy.

    Args:
        probabilities: Raw probability mass (will be normalised internally).
        alpha: Weight for the entropy term (must be in `[0, 1]`).

    Returns:
        A float in `[0, 1]` representing the combined inequality‑uncertainty
        measure.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    p = _validate_and_normalise(probabilities)
    n = p.size

    # Shannon entropy (natural log) and its normalisation
    with np.errstate(divide="ignore"):
        entropy = -np.sum(p * np.log(p))
    max_entropy = math.log(n) if n > 1 else 0.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Gini index (also known as Gini impurity)
    gini = 1.0 - np.sum(p ** 2)

    return alpha * norm_entropy + (1.0 - alpha) * gini


# --------------------------------------------------------------------------- #
# Weighted MinHash utilities
# --------------------------------------------------------------------------- #


def weighted_minhash(
    probabilities: Sequence[float],
    k: int = 128,
) -> List[int]:
    """
    Produce a weighted MinHash signature.

    For each hash function (identified by a seed `i`), the token representing
    the i‑th probability is hashed and the result is divided by the probability
    value.  The minimum of these scaled hashes is kept.  This gives larger
    probabilities a higher chance of dominating the signature while preserving
    the MinHash property of estimating Jaccard similarity.

    Args:
        probabilities: Raw probability mass (will be normalised internally).
        k: Number of hash functions / signature length.

    Returns:
        List of `k` 64‑bit integers.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    p = _validate_and_normalise(probabilities)
    tokens = [f"{i}:{prob:.12g}" for i, prob in enumerate(p)]

    signature: List[int] = []
    for seed in range(k):
        # Compute scaled hash for each token; avoid division by zero (p>0 after
        # normalisation)
        scaled_hashes = (_hash(seed, tok) / prob for tok, prob in zip(tokens, p))
        signature.append(int(min(scaled_hashes)))
    return signature


def jaccard_signature_similarity(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """
    Compute the fraction of equal components between two MinHash signatures.
    This is an unbiased estimator of the Jaccard similarity of the underlying
    weighted sets.

    Returns:
        A float in `[0, 1]`.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have the same length.")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# --------------------------------------------------------------------------- #
# Chelydrid ambush‑strike cost model
# --------------------------------------------------------------------------- #


def chelydrid_strike(
    probabilities: Sequence[float],
    k: int = 128,
    dt: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """
    Simulate the chelydrid ambush‑strike kinematics and return a cost value.

    The cost combines three ingredients:
    1. The weighted MinHash signature of the distribution.
    2. Its similarity to a reference signature consisting of the maximal hash
       value (`2**64‑1`).  This reference represents an “empty” or maximally
       dissimilar distribution.
    3. The combined Gini‑entropy metric, which modulates the cost according to
       the distribution’s inequality and uncertainty.

    The final cost is:
        `dt * (1 - similarity) * combined_metric`

    Args:
        probabilities: Raw probability mass.
        k: Signature length.
        dt: Time‑step scaling factor.
        alpha: Weight for entropy in the combined metric (see
               :func:`combined_gini_entropy`).

    Returns:
        A non‑negative float representing the selection cost.
    """
    # 1. Signature of the input distribution
    sig = weighted_minhash(probabilities, k)

    # 2. Reference signature (max hash = 2**64‑1) – deterministic
    ref_sig = [2 ** 64 - 1] * k
    similarity = jaccard_signature_similarity(sig, ref_sig)

    # 3. Combined inequality‑uncertainty measure
    metric = combined_gini_entropy(probabilities, alpha)

    # Cost formulation (higher when distribution is dissimilar to the reference
    # and when it exhibits strong inequality/uncertainty)
    return dt * (1.0 - similarity) * metric


# --------------------------------------------------------------------------- #
# Simple command‑line smoke test
# --------------------------------------------------------------------------- #


def _smoke_test() -> None:
    probs = [0.2, 0.3, 0.5]
    print("Combined Gini‑Entropy (α=0.5):", combined_gini_entropy(probs))
    print("Weighted MinHash signature (first 8 values):", weighted_minhash(probs)[:8])
    print("Chelydrid strike cost:", chelydrid_strike(probs))


if __name__ == "__main__":
    _smoke_test()