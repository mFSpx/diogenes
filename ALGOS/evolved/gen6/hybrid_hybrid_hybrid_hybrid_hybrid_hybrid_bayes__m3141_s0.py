# DARWIN HAMMER — match 3141, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hdc_hy_m2007_s0.py (gen5)
# parent_b: hybrid_hybrid_bayes_update__hybrid_hybrid_hybrid_m1675_s1.py (gen5)
# born: 2026-05-29T23:48:03Z

"""
Hybrid Fusion Module

Parents:
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s1 (Bayesian + HDC)
- hybrid_hybrid_bayes_update__hybrid_hybrid_hybrid_m1675_s1 (Gaussian Bayes + Geometric Algebra)

Mathematical Bridge
------------------
The Bayesian scalar confidence *p* (posterior probability) is combined with a
high‑dimensional contextual affinity *s* obtained from Hyperdimensional
Computing (HDC).  In the second parent, similarity is expressed through the
inner product of grade‑1 multivectors (vectors) and uncertainty is modelled by
a Gaussian kernel with variance σ².

We map the bipolar HDC vector **v** ∈ {‑1,+1}ᴰ to a grade‑1 multivector
`MV(v) = Σ_i v_i e_i`.  The normalized dot product used in the HDC parent
is exactly the GA inner product divided by the dimension *D*:

    s = (v · r) / D = ⟨ MV(v), MV(r) ⟩ / D   ∈ [‑1, 1]

A Gaussian weighting of this similarity,

    w = exp( - (1‑s)² / (2σ²) ),

modulates the Bayesian posterior *p*.  The fused decision metric is

    score = p * w

which reduces to the original HDC‑Bayesian fusion when σ → ∞
(`w → 1`).  The functions below implement this unified pipeline:
1. `bayesian_gaussian_posterior` – Bayesian update with a Gaussian likelihood.
2. `features_to_hdc_vector` – deterministic construction of a bipolar HDC vector
   from integer feature counts.
3. `hdc_vector_to_multivector` – conversion of an HDC vector to a grade‑1 GA
   multivector (dictionary mapping frozenset indices to scalar weights).
4. `multivector_inner` – GA inner product for grade‑1 multivectors.
5. `hybrid_gaussian_hdc_score` – final fused score using the bridge described
   above.

All utilities are self‑contained and rely only on `numpy`, `math`, `random`,
`sys`, and `pathlib`.
"""

import math
import random
import sys
from pathlib import Path
import hashlib
import numpy as np
from typing import List, Tuple, Dict, FrozenSet

# ----------------------------------------------------------------------
# Deterministic pseudo‑random generator seeded from a string
# ----------------------------------------------------------------------
def _seed_from_string(seed_str: str) -> random.Random:
    """Create a Random instance seeded deterministically from *seed_str*."""
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    # Use first 8 bytes as a 64‑bit integer seed
    seed = int.from_bytes(h[:8], "big", signed=False)
    return random.Random(seed)


# ----------------------------------------------------------------------
# Parent A utilities (phash, hamming)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Compute a 64‑bit binary hash based on the average of *values*."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit‑strings."""
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# Hyperdimensional Computing (HDC) – feature to vector
# ----------------------------------------------------------------------
def features_to_hdc_vector(
    counts: List[int],
    dim: int = 10000,
    seed_str: str = "hdc_seed",
) -> np.ndarray:
    """
    Build a bipolar HDC vector from integer *counts*.

    Each dimension has a deterministic symbol vector (±1) generated from the
    seed string combined with the dimension index.  The final vector is the
    weighted super‑position of those symbols, followed by bipolarization.
    """
    rng = _seed_from_string(seed_str)
    # Pre‑generate deterministic symbol vectors (+1 / -1) for each dimension
    symbols = rng.choices([-1, 1], k=dim)

    # Weighted sum
    vec = np.zeros(dim, dtype=np.float32)
    for idx, cnt in enumerate(counts):
        if idx >= dim:
            break
        vec += cnt * symbols[idx]

    # Bipolarize: sign(vec) → {‑1, +1}
    bipolar = np.where(vec >= 0, 1, -1).astype(np.int8)
    return bipolar


# ----------------------------------------------------------------------
# Geometric Algebra core (grade‑1 multivectors)
# ----------------------------------------------------------------------
def hdc_vector_to_multivector(vec: np.ndarray) -> Dict[FrozenSet[int], float]:
    """
    Convert a bipolar HDC vector to a grade‑1 multivector.

    The multivector is represented as a dict mapping a frozenset containing the
    basis index to its scalar weight (±1).
    """
    mv: Dict[FrozenSet[int], float] = {}
    for i, val in enumerate(vec):
        if val != 0:
            mv[frozenset({i})] = float(val)
    return mv


def multivector_inner(
    mv_a: Dict[FrozenSet[int], float],
    mv_b: Dict[FrozenSet[int], float],
) -> float:
    """
    Inner product ⟨A, B⟩ for grade‑1 multivectors.

    Since both are grade‑1, the inner product reduces to the sum of products
    of matching basis blades.
    """
    total = 0.0
    # Intersection of basis blades
    for blade, a_val in mv_a.items():
        b_val = mv_b.get(blade)
        if b_val is not None:
            total += a_val * b_val
    return total


# ----------------------------------------------------------------------
# Bayesian Gaussian update (Parent B)
# ----------------------------------------------------------------------
def bayesian_gaussian_posterior(
    prior: float,
    likelihood: float,
    sigma: float,
) -> float:
    """
    Perform a Bayesian update where the likelihood is modelled by a Gaussian
    centred at 1.0 (perfect match).  The posterior (unnormalised) is

        posterior ∝ prior * exp( - (likelihood‑1)² / (2σ²) ).

    The function returns the normalised posterior assuming a uniform prior
    over [0,1]; the normalisation factor is omitted because it cancels in
    downstream scoring.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    exponent = -((likelihood - 1.0) ** 2) / (2.0 * sigma ** 2)
    return prior * math.exp(exponent)


# ----------------------------------------------------------------------
# Fusion: Gaussian‑weighted HDC similarity modulating the Bayesian posterior
# ----------------------------------------------------------------------
def hybrid_gaussian_hdc_score(
    prior: float,
    feature_counts: List[int],
    reference_vector: np.ndarray,
    sigma: float = 0.2,
    dim: int = 10000,
    seed_str: str = "hdc_seed",
) -> float:
    """
    Compute the fused decision score.

    Steps
    -----
    1. Build a bipolar HDC vector from *feature_counts*.
    2. Convert both the query and *reference_vector* to grade‑1 multivectors.
    3. Compute the normalized HDC similarity `s ∈ [‑1, 1]` via GA inner product.
    4. Treat `s` as a likelihood and obtain a Gaussian‑adjusted posterior.
    5. Return `score = posterior`.

    The returned value lies in [0, 1] for priors in that range.
    """
    # 1. HDC vector from features
    query_vec = features_to_hdc_vector(feature_counts, dim=dim, seed_str=seed_str)

    # Ensure reference vector has correct shape and bipolar values
    if reference_vector.shape != (dim,):
        raise ValueError(f"reference_vector must have shape ({dim},)")
    ref_vec = np.where(reference_vector >= 0, 1, -1).astype(np.int8)

    # 2. Convert to multivectors
    mv_query = hdc_vector_to_multivector(query_vec)
    mv_ref = hdc_vector_to_multivector(ref_vec)

    # 3. GA inner product (equivalent to dot product)
    raw_inner = multivector_inner(mv_query, mv_ref)  # integer in [‑dim, dim]
    s = raw_inner / dim  # normalized similarity ∈ [‑1, 1]

    # 4. Gaussian‑adjusted Bayesian posterior
    posterior = bayesian_gaussian_posterior(prior, s, sigma)

    # 5. Clamp to [0,1] for safety
    return max(0.0, min(1.0, posterior))


# ----------------------------------------------------------------------
# Additional demonstration function: combine HDC hash with GA similarity
# ----------------------------------------------------------------------
def hybrid_decision_score(
    prior: float,
    feature_counts: List[int],
    reference_vector: np.ndarray,
    sigma: float = 0.2,
    dim: int = 10000,
    seed_str: str = "hdc_seed",
) -> float:
    """
    Variant of `hybrid_gaussian_hdc_score` that also incorporates the binary
    perceptual hash of the feature‑derived HDC vector.  The hash distance is
    turned into a similarity factor `h ∈ [0,1]` and multiplied with the
    Gaussian‑adjusted posterior.

    This showcases a true hybrid of all three cores:
    * Bayesian Gaussian update (Parent B)
    * HDC vector construction & similarity (Parent A)
    * Binary hash utility (Parent A)
    """
    # Compute base score
    base_score = hybrid_gaussian_hdc_score(
        prior, feature_counts, reference_vector, sigma, dim, seed_str
    )

    # Build HDC vector and its phash
    query_vec = features_to_hdc_vector(feature_counts, dim=dim, seed_str=seed_str)
    phash = compute_phash(query_vec.tolist())

    # Reference hash: deterministic from reference vector
    ref_phash = compute_phash(reference_vector.tolist())

    # Hamming similarity
    ham_dist = hamming_distance(phash, ref_phash)
    h = 1.0 - ham_dist / 64.0  # 64‑bit hash -> similarity in [0,1]

    return base_score * h


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic example
    D = 10000
    # Random reference vector (bipolar)
    rng = random.Random(42)
    ref = np.array([rng.choice([-1, 1]) for _ in range(D)], dtype=np.int8)

    # Feature counts (e.g., word frequencies)
    features = [5, 2, 0, 3, 7] + [0] * (D - 5)

    prior_prob = 0.6
    sigma_val = 0.15

    score = hybrid_gaussian_hdc_score(
        prior=prior_prob,
        feature_counts=features,
        reference_vector=ref,
        sigma=sigma_val,
        dim=D,
        seed_str="example_seed",
    )
    print(f"Hybrid Gaussian‑HDC score: {score:.6f}")

    decision = hybrid_decision_score(
        prior=prior_prob,
        feature_counts=features,
        reference_vector=ref,
        sigma=sigma_val,
        dim=D,
        seed_str="example_seed",
    )
    print(f"Hybrid decision score (with hash): {decision:.6f}")