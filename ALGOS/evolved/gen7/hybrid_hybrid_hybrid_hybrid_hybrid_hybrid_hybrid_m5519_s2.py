# DARWIN HAMMER — match 5519, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1995_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1908_s0.py (gen6)
# born: 2026-05-30T00:02:29Z

"""Hybrid Morphology‑MinHash‑NLMS Graph Weighting

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – provides geometric morphology (sphericity and flatness) and a
  MinHash‑style signature generator whose noise schedule is driven by a
  diffusion‑forcing prior.
* **Parent B** – builds graph edge weights from a Bayesian marginal `m(i,j)` and
  an NLMS‑derived scalar factor `Δ_{ij} = (1‑μ)·(1‑m(i,j))`.

**Mathematical bridge** – The morphology‑derived prior `π` (a function of the
sphericity and flatness indices) is used in two places:

1. It scales the diffusion noise injected into the MinHash signature, thus
   shaping the similarity distribution between node signatures.
2. It modulates the NLMS step‑size `μ` for each node, producing a
   node‑specific factor `μ_i = μ_global·π_i`.  The edge‑wise NLMS factor
   `Δ_{ij}` then uses the average of the two node‑specific step‑sizes.

The resulting edge weight is


w_{ij} = d(i,j)·Δ_{ij} + ε,
Δ_{ij} = (1‑μ̄_{ij})·(1‑m(i,j)),
μ̄_{ij} = (μ_i + μ_j)/2,
m(i,j) = prior_similarity(π_i,π_j)·signature_overlap(sig_i, sig_j)


where `d(i,j)` is a physical distance, `ε` a small jitter, and
`signature_overlap` counts matching MinHash bins.

The three public functions below illustrate the hybrid workflow.
"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Morphology utilities (from Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def sphericity(self) -> float:
        """Sphericity index = cubic root of volume divided by length."""
        if min(self.length, self.width, self.height) <= 0:
            raise ValueError("dimensions must be positive")
        return (self.length * self.width * self.height) ** (1.0 / 3.0) / self.length

    def flatness(self) -> float:
        """Flatness index = (length+width)/(2·height)."""
        if min(self.length, self.width, self.height) <= 0:
            raise ValueError("dimensions must be positive")
        return (self.length + self.width) / (2.0 * self.height)


def shape_prior(morph: Morphology) -> float:
    """
    Compute a scalar prior π ∈ (0,1] from morphology.

    The prior is a normalized blend of sphericity and flatness:
        π = 0.6·sphericity_norm + 0.4·flatness_norm
    where each component is clipped to [0,1].

    The prior will later scale diffusion noise and NLMS step‑size.
    """
    sph = morph.sphericity()
    flat = morph.flatness()
    # Normalise heuristically (empirical caps)
    sph_n = min(max(sph, 0.0), 1.0)
    flat_n = min(max(flat / 3.0, 0.0), 1.0)  # flatness typically larger; scale down
    prior = 0.6 * sph_n + 0.4 * flat_n
    return max(min(prior, 1.0), 0.01)  # avoid exact zero


# ----------------------------------------------------------------------
# MinHash‑style signature with diffusion forcing (Parent A)
# ----------------------------------------------------------------------


def _hash_int(seed: int, token: str) -> int:
    """Deterministic 32‑bit integer hash."""
    h = hashlib.blake2b(digest_size=4)
    h.update(seed.to_bytes(4, "big"))
    h.update(token.encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def minhash_signature(sequence: str, seed: int, n_hashes: int, prior: float) -> List[int]:
    """
    Generate a MinHash signature of length `n_hashes`.

    Diffusion forcing: before hashing each token we add Gaussian noise
    scaled by (1‑π) to the token's ordinal value, then cast back to a string.
    This makes the signature distribution sensitive to the morphology prior.
    """
    tokens = sequence.split()
    signature = [2 ** 31 - 1] * n_hashes  # initialise with max int
    for token in tokens:
        # base ordinal (simple deterministic mapping)
        base = sum(ord(c) for c in token) % (2 ** 16)
        # diffusion noise
        noise = random.gauss(0, 1.0) * (1.0 - prior)
        noisy_val = int(base + noise) & 0xFFFF
        noisy_token = f"{noisy_val}"
        for i in range(n_hashes):
            h = _hash_int(seed + i, noisy_token)
            if h < signature[i]:
                signature[i] = h
    return signature


def signature_overlap(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Jaccard‑like overlap for fixed‑size MinHash signatures:
        overlap = (# equal bins) / n_hashes
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Bayesian marginal and NLMS factor (Parent B)
# ----------------------------------------------------------------------


def bayesian_marginal(prior_i: float, prior_j: float, overlap: float) -> float:
    """
    Simple Bayesian marginal m(i,j) combining morphology priors and
    signature similarity.

    We treat the priors as independent Bernoulli probabilities and use
    overlap as likelihood:

        m = (π_i * π_j) * overlap
    """
    return (prior_i * prior_j) * overlap


def nlms_factor(mu_global: float, prior_i: float, prior_j: float, marginal: float) -> float:
    """
    Compute the NLMS‑derived scalar Δ_{ij}.

    Node‑specific step‑sizes:
        μ_i = μ_global * π_i
        μ_j = μ_global * π_j
    Average μ̄ = (μ_i + μ_j)/2

    Then:
        Δ = (1 - μ̄) * (1 - m)
    """
    mu_i = mu_global * prior_i
    mu_j = mu_global * prior_j
    mu_bar = (mu_i + mu_j) / 2.0
    delta = (1.0 - mu_bar) * (1.0 - marginal)
    return max(delta, 0.0)  # ensure non‑negative


# ----------------------------------------------------------------------
# Hybrid edge weight computation
# ----------------------------------------------------------------------


def compute_edge_weight(
    seq_i: str,
    seq_j: str,
    morph_i: Morphology,
    morph_j: Morphology,
    distance: float,
    seed: int = 42,
    n_hashes: int = 64,
    mu_global: float = 0.1,
    epsilon: float = 1e-6,
) -> float:
    """
    Unified hybrid operation:

    1. Convert morphologies to priors π_i, π_j.
    2. Generate MinHash signatures with diffusion forcing.
    3. Compute signature overlap and Bayesian marginal m(i,j).
    4. Derive NLMS factor Δ_{ij}.
    5. Return final edge weight w_{ij}.
    """
    # 1. Priors
    pi_i = shape_prior(morph_i)
    pi_j = shape_prior(morph_j)

    # 2. Signatures
    sig_i = minhash_signature(seq_i, seed, n_hashes, pi_i)
    sig_j = minhash_signature(seq_j, seed, n_hashes, pi_j)

    # 3. Overlap & marginal
    overlap = signature_overlap(sig_i, sig_j)
    m_ij = bayesian_marginal(pi_i, pi_j, overlap)

    # 4. NLMS factor
    delta = nlms_factor(mu_global, pi_i, pi_j, m_ij)

    # 5. Edge weight
    weight = distance * delta + epsilon
    return weight


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define two example nodes
    morph_a = Morphology(length=2.0, width=1.5, height=1.0, mass=3.2)
    morph_b = Morphology(length=1.8, width=1.4, height=0.9, mass=2.9)

    seq_a = "the quick brown fox jumps over the lazy dog"
    seq_b = "the quick brown dog leaps over the lazy fox"

    # Physical distance (arbitrary)
    phys_dist = 5.0

    w = compute_edge_weight(
        seq_i=seq_a,
        seq_j=seq_b,
        morph_i=morph_a,
        morph_j=morph_b,
        distance=phys_dist,
        seed=123,
        n_hashes=32,
        mu_global=0.15,
    )

    print(f"Hybrid edge weight: {w:.6f}")
    # Verify that weight is a finite positive number
    assert math.isfinite(w) and w > 0, "Edge weight computation failed"