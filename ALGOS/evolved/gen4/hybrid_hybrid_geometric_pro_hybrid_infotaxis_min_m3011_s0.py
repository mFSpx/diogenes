# DARWIN HAMMER — match 3011, survivor 0
# gen: 4
# parent_a: hybrid_geometric_product_hybrid_hybrid_fisher_m1171_s3.py (gen3)
# parent_b: hybrid_infotaxis_minhash_m63_s5.py (gen1)
# born: 2026-05-29T23:47:13Z

"""Hybrid MinHash‑Geometric‑Fisher (HMGF)

This module fuses the core mathematics of two parents:

* **Parent A** – a Clifford geometric product implementation that treats
  scalars, vectors (grade‑1 blades) and higher‑grade blades as elements of a
  multivector algebra.

* **Parent B** – a MinHash/Jaccard similarity engine together with Shannon‑
  entropy utilities.  The entropy of a token‑count histogram can be turned
  into a Fisher‑information‑like scalar (high entropy ⇒ low information).

**Mathematical bridge**

Both frameworks ultimately produce *scalar* quantities:

* The MinHash similarity `s ∈ [0,1]` and the entropy‑derived Fisher term
  `I ≥ 0` are ordinary real numbers.
* In a Clifford algebra a scalar is the grade‑0 blade `∅`.  By embedding
  `s` and `I` as coefficients of basis blades we obtain multivectors
  `M_s` and `M_I`.  The geometric product `M_s * M_I` automatically
  combines the two scalars (inner product) while preserving any
  directional (grade‑1) information that may be attached to the signatures.

The three public functions below demonstrate this fusion:

1. `signature_to_multivector` – builds a grade‑1 multivector from a MinHash
   signature, using each hash component as a basis index.
2. `entropy_fisher_information` – converts a raw count vector into a scalar
   Fisher‑information estimate via Shannon entropy.
3. `hybrid_cost` – multiplies the two multivectors with the geometric product,
   extracts the scalar part and returns a single “hybrid cost’’ that blends
   deterministic similarity, uncertainty and probabilistic belief.

All code is self‑contained and relies only on the standard library,
`numpy`, `math`, `random`, `sys` and `pathlib`."""

from __future__ import annotations

import hashlib
import math
import random
import sys
import pathlib
from typing import Dict, Iterable, List, Tuple, FrozenSet, Set

import numpy as np

# ----------------------------------------------------------------------
# MinHash utilities (from Parent B)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
DEFAULT_K = 128
_EPS = 1e-12


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash for a (seed, token) pair."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = DEFAULT_K) -> List[int]:
    """Compute a MinHash signature of length *k* for the given token collection."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    token_set: Set[str] = {t for t in tokens if t}
    if not token_set:
        return [MAX64] * k
    return [min(_hash(i, t) for t in token_set) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via the fraction of equal components."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def _normalize(probs: List[float]) -> List[float]:
    total = sum(probs)
    if total <= 0:
        raise ValueError("probability mass must be positive")
    return [p / total for p in probs]


def entropy_from_counts(counts: List[int]) -> float:
    """Shannon entropy of a discrete distribution defined by raw counts."""
    if not counts:
        raise ValueError("counts must not be empty")
    probs = _normalize([float(c) for c in counts])
    return -sum(p * math.log(max(p, _EPS)) for p in probs)


# ----------------------------------------------------------------------
# Clifford geometric algebra helpers (from Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """
    Return a sorted list of indices and the sign (+1 or -1) required to
    reorder the original list into that sorted order, cancelling duplicate
    indices (e_i * e_i = 1).
    """
    # Work on a mutable copy
    lst = list(indices)
    sign = 1

    # Bubble‑sort while tracking swaps
    n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1

    # Cancel duplicate indices (they square to +1)
    i = 0
    while i < len(lst) - 1:
        if lst[i] == lst[i + 1]:
            del lst[i : i + 2]  # remove the pair
            # No sign change because e_i*e_i = 1 (scalar)
        else:
            i += 1

    return lst, sign


def geometric_product(mv1: Dict[FrozenSet[int], float],
                      mv2: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """
    Compute the geometric product of two multivectors.
    Multivectors are represented as dictionaries:
        blade (frozenset of basis indices) -> coefficient (float)
    The empty frozenset represents the scalar (grade‑0) blade.
    """
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv1.items():
        for blade_b, coeff_b in mv2.items():
            # concatenate the index lists
            combined = list(blade_a) + list(blade_b)
            sorted_idxs, sign = _blade_sign(combined)
            new_blade = frozenset(sorted_idxs)
            result[new_blade] = result.get(new_blade, 0.0) + sign * coeff_a * coeff_b
    return result


def scalar_part(mv: Dict[FrozenSet[int], float]) -> float:
    """Extract the grade‑0 (scalar) component of a multivector."""
    return mv.get(frozenset(), 0.0)


# ----------------------------------------------------------------------
# Hybrid constructions
# ----------------------------------------------------------------------
def signature_to_multivector(sig: List[int],
                             weight: float = 1.0,
                             base_offset: int = 0) -> Dict[FrozenSet[int], float]:
    """
    Embed a MinHash signature into a grade‑1 multivector.

    Each hash component `h_i` becomes a basis vector `e_{base_offset+i}`.
    The coefficient attached to each basis blade is `weight * h_i / MAX64`,
    i.e. a normalized value in [0, weight].

    The resulting multivector is a *sum* of grade‑1 blades.
    """
    mv: Dict[FrozenSet[int], float] = {}
    for i, h in enumerate(sig):
        idx = base_offset + i
        coeff = weight * (h / MAX64)
        blade = frozenset([idx])
        mv[blade] = mv.get(blade, 0.0) + coeff
    return mv


def entropy_fisher_information(counts: List[int]) -> float:
    """
    Convert a raw count histogram into a scalar Fisher‑information estimate.

    We use the inverse of (entropy + ε) as a simple proxy:
        I = 1 / (H + ε)

    Higher entropy ⇒ lower information, matching the intuition behind Fisher
    information for a Gaussian with variance proportional to entropy.
    """
    H = entropy_from_counts(counts)
    return 1.0 / (H + _EPS)


def hybrid_cost(sig_a: List[int],
                sig_b: List[int],
                counts_a: List[int],
                counts_b: List[int],
                base_offset: int = 0) -> float:
    """
    Compute a unified hybrid cost between two objects.

    1. Build multivectors `M_a` and `M_b` from the MinHash signatures.
    2. Multiply them with the geometric product.
    3. Extract the scalar part and weight it by the product of the two
       Fisher‑information scalars derived from the count histograms.
    4. Add the raw MinHash Jaccard similarity as a linear term.

    The final scalar blends deterministic similarity, uncertainty and
    probabilistic belief into a single number.
    """
    # Step 1 – embed signatures
    mv_a = signature_to_multivector(sig_a, weight=1.0, base_offset=base_offset)
    mv_b = signature_to_multivector(sig_b, weight=1.0, base_offset=base_offset)

    # Step 2 – geometric product
    prod_mv = geometric_product(mv_a, mv_b)

    # Step 3 – scalar extraction and Fisher weighting
    scalar = scalar_part(prod_mv)
    fisher_a = entropy_fisher_information(counts_a)
    fisher_b = entropy_fisher_information(counts_b)

    weighted_scalar = scalar * fisher_a * fisher_b

    # Step 4 – add raw Jaccard similarity
    jaccard = similarity(sig_a, sig_b)
    return weighted_scalar + jaccard


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def hybrid_multivector_product(sig_a: List[int],
                               sig_b: List[int],
                               base_offset: int = 0) -> Dict[FrozenSet[int], float]:
    """
    Return the full geometric product multivector of two signature‑derived
    multivectors.  Useful for inspecting higher‑grade components.
    """
    mv_a = signature_to_multivector(sig_a, weight=1.0, base_offset=base_offset)
    mv_b = signature_to_multivector(sig_b, weight=1.0, base_offset=base_offset)
    return geometric_product(mv_a, mv_b)


def hybrid_similarity_with_fisher(sig_a: List[int],
                                  sig_b: List[int],
                                  counts_a: List[int],
                                  counts_b: List[int]) -> float:
    """
    A lightweight variant that returns only the Jaccard similarity scaled by
    the geometric‑product scalar (without the additive Jaccard term).
    """
    prod_mv = hybrid_multivector_product(sig_a, sig_b)
    scalar = scalar_part(prod_mv)
    fisher = entropy_fisher_information(counts_a) * entropy_fisher_information(counts_b)
    return scalar * fisher


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two random token collections
    vocab = [f"word{i}" for i in range(1000)]
    tokens_a = random.sample(vocab, 120)
    tokens_b = random.sample(vocab, 150)

    # MinHash signatures
    sig_a = signature(tokens_a, k=64)
    sig_b = signature(tokens_b, k=64)

    # Simulated count histograms (e.g., term frequencies)
    counts_a = [random.randint(0, 20) for _ in range(20)]
    counts_b = [random.randint(0, 20) for _ in range(20)]

    # Compute hybrid cost
    cost = hybrid_cost(sig_a, sig_b, counts_a, counts_b, base_offset=0)
    print(f"Hybrid cost (scalar + Jaccard): {cost:.6f}")

    # Show full product multivector (only non‑zero blades)
    prod_mv = hybrid_multivector_product(sig_a, sig_b)
    nonzero = {blade: coeff for blade, coeff in prod_mv.items() if abs(coeff) > 1e-12}
    print(f"Geometric product non‑zero blades: {len(nonzero)}")
    # Print a few example blades
    for i, (blade, coeff) in enumerate(nonzero.items()):
        if i >= 5:
            break
        print(f"  Blade {sorted(blade)} -> {coeff:.4e}")

    # Verify that the scalar part matches expectation
    scalar = scalar_part(prod_mv)
    print(f"Scalar part of product: {scalar:.6f}")