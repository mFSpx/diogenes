# DARWIN HAMMER — match 3074, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1818_s4.py (gen5)
# parent_b: hybrid_fisher_localization_hybrid_hybrid_hybrid_m1503_s4.py (gen6)
# born: 2026-05-29T23:47:39Z

"""Hybrid Morphology‑Fisher Allocation

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1818_s4 (MinHash‑HD morphology
  with decision‑hygiene weighting)
- hybrid_fisher_localization_hybrid_hybrid_hybrid_m1503_s4 (Gaussian beam,
  Fisher information, Bayesian update for allocation)

Mathematical bridge
------------------
The morphology pipeline yields a bound hyper‑vector **B** = **M** ⊙ **H**.
Its cosine similarity to a goal vector **G** (cosθ) is a value in [‑1, 1]
that we reinterpret as a *prior* probability *p₀* for the Bayesian update
used in the Fisher‑localization pipeline.

Conversely, the decision‑hygiene score *Sₕ* (entropy‑weighted linear
combination of feature counts) provides a *likelihood scaling* that modulates
the Fisher information term.

The fused system therefore computes

    p_prior   = (cosθ + 1) / 2                     # map to [0,1]
    p_alloc   = GaussianBeam(t, μ = p_prior, σ)   # allocation probability
    I_fisher  = FisherScore(t, μ = p_prior, σ)
    L_likelihood = Sₕ / Sₕ_max                     # normalised hygiene score
    p_post    = BayesianUpdate(p_alloc, I_fisher, L_likelihood)

The final unified metric is the posterior probability *p_post* multiplied by
the original cosine similarity to retain the geometric contribution.

The module implements this fused pipeline with three public functions:
    • morphology_vector – builds **M** from a Morphology instance.
    • decision_hygiene_score – reproduces the Parent A hygiene metric.
    • hybrid_allocation_score – combines both parents into a single score.
"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
K = 128                     # MinHash size
H_MAX = math.log2(9)        # entropy normaliser from Parent A


# ----------------------------------------------------------------------
# Parent A – morphology & MinHash utilities
# ----------------------------------------------------------------------
Vector = List[float]


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float
    tokens: List[str]


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = K) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    sig = []
    for seed in range(k):
        sig.append(min(_hash(seed, t) for t in toks))
    return sig


def normalize_signature(sig: List[int]) -> np.ndarray:
    """Map integer signature to real‑valued vector in [0,1]."""
    arr = np.array(sig, dtype=np.float64)
    return arr / MAX64


def morphology_vector(morph: Morphology) -> np.ndarray:
    """
    Build a length‑K hyper‑vector **M** from a Morphology.
    The first four slots store the geometric attributes, the next slot stores
    the token count, and the remaining slots are zero‑filled.
    """
    base = np.zeros(K, dtype=np.float64)
    base[0] = morph.length
    base[1] = morph.width
    base[2] = morph.height
    base[3] = morph.mass
    base[4] = len(morph.tokens)
    return base


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity handling zero vectors safely."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ----------------------------------------------------------------------
# Parent A – decision hygiene utilities
# ----------------------------------------------------------------------
def decision_hygiene_score(
    counts: List[int],
    w_pos: List[float],
    w_neg: List[float],
) -> float:
    """
    Compute the hybrid hygiene score Sₕ.

    s = w⁺·v − w⁻·v
    p = v / Σv
    H = − Σ p·log₂ p
    Sₕ = s·(1 + H / H_MAX)
    """
    if len(counts) != len(w_pos) or len(counts) != len(w_neg):
        raise ValueError("Length mismatch among counts and weight vectors")
    v = np.array(counts, dtype=np.float64)
    w_p = np.array(w_pos, dtype=np.float64)
    w_n = np.array(w_neg, dtype=np.float64)

    s = float(np.dot(w_p, v) - np.dot(w_n, v))
    total = v.sum()
    if total == 0:
        H = 0.0
    else:
        p = v / total
        # avoid log2(0) by masking zeros
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log2(p[mask])))
    return s * (1.0 + H / H_MAX)


# ----------------------------------------------------------------------
# Parent B – Fisher localisation utilities
# ----------------------------------------------------------------------
@dataclass
class Entity:
    timestamp: float
    spatial_load: float
    privacy_load: float


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian probability density (unnormalised) centered at *center*."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian model."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def bayesian_update(probability: float, likelihood: float, prior: float) -> float:
    """Standard Bayesian update with scalar arguments."""
    numerator = probability * likelihood
    denominator = numerator + (1.0 - probability) * (1.0 - prior)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def vram_allocation_planning(entity: Entity, allocation_probability: float) -> float:
    """Resource‑aware allocation metric from Parent B."""
    return entity.spatial_load * allocation_probability / (entity.privacy_load + 1.0)


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_allocation_score(
    morph: Morphology,
    entity: Entity,
    goal_vector: np.ndarray,
    counts: List[int],
    w_pos: List[float],
    w_neg: List[float],
    width_factor: float = 0.1,
) -> float:
    """
    Unified metric that merges the hyper‑dimensional MinHash similarity
    with Fisher‑information‑based allocation.

    Steps
    -----
    1. Build **M**, obtain **H**, bind to get **B**.
    2. Compute cosine similarity cosθ between **B** and *goal_vector*.
    3. Map cosθ → prior probability p₀ = (cosθ + 1)/2.
    4. Compute decision‑hygiene score Sₕ and normalise it to a likelihood L.
    5. Use p₀ as the Gaussian centre; width = width_factor·(1‑entropy/H_MAX).
    6. Evaluate allocation probability (Gaussian) and Fisher information.
    7. Bayesian update with (allocation_probability, Fisher information, L).
    8. Return posterior multiplied by original cosine similarity to retain geometric influence.
    """
    # 1. Hyper‑dimensional representation
    M = morphology_vector(morph)                     # shape (K,)
    H_int = minhash_signature(morph.tokens, K)       # list[int]
    H = normalize_signature(H_int)                  # shape (K,)
    B = M * H                                        # element‑wise binding

    # 2. Cosine similarity
    cos_theta = cosine_similarity(B, goal_vector)

    # 3. Prior probability from similarity
    p_prior = (cos_theta + 1.0) / 2.0                 # ∈ [0,1]

    # 4. Decision hygiene score and normalisation
    S_h = decision_hygiene_score(counts, w_pos, w_neg)
    # Normalise by an empirical bound (here we use absolute max possible with given weights)
    max_abs_score = max(abs(S_h), 1e-12)             # avoid division by zero
    L_likelihood = (S_h / max_abs_score + 1.0) / 2.0  # map to [0,1]

    # 5. Width derived from entropy (higher entropy → narrower focus)
    v = np.array(counts, dtype=np.float64)
    total = v.sum()
    if total == 0:
        entropy = 0.0
    else:
        p = v / total
        mask = p > 0
        entropy = -float(np.sum(p[mask] * np.log2(p[mask])))
    width = max(width_factor * (1.0 - entropy / H_MAX), 1e-6)

    # 6. Allocation probability and Fisher information
    alloc_prob = gaussian_beam(entity.timestamp, p_prior, width)
    I_fisher = fisher_score(entity.timestamp, p_prior, width)

    # 7. Bayesian posterior
    posterior = bayesian_update(alloc_prob, I_fisher, L_likelihood)

    # 8. Final fused score
    final_score = posterior * cos_theta
    return final_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample morphology
    sample_morph = Morphology(
        length=12.4,
        width=5.7,
        height=3.2,
        mass=8.9,
        tokens=["alpha", "beta", "gamma", "delta", "epsilon"]
    )

    # Goal vector (unit vector for simplicity)
    goal = np.ones(K, dtype=np.float64)
    goal /= np.linalg.norm(goal)

    # Sample entity
    sample_entity = Entity(
        timestamp=42.0,
        spatial_load=3.5,
        privacy_load=0.8
    )

    # Feature counts and weights for hygiene score
    counts = [random.randint(0, 10) for _ in range(9)]
    w_pos = [random.uniform(0.1, 1.0) for _ in range(9)]
    w_neg = [random.uniform(0.0, 0.5) for _ in range(9)]

    score = hybrid_allocation_score(
        morph=sample_morph,
        entity=sample_entity,
        goal_vector=goal,
        counts=counts,
        w_pos=w_pos,
        w_neg=w_neg,
    )
    print(f"Hybrid allocation score: {score:.6f}")