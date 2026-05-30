# DARWIN HAMMER — match 1580, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m759_s0.py (gen5)
# parent_b: hybrid_hybrid_label_foundry_hybrid_hybrid_minimu_m542_s0.py (gen3)
# born: 2026-05-29T23:37:39Z

"""Hybrid Algorithm Fusion of Clifford Algebra and Bayesian Labeling

Parents:
- Parent A: `hybrid_hybrid_hybrid_geomet_hybrid_hybrid_model__m176_s1.py` – provides
  Clifford (geometric) algebra operations on multivectors.
- Parent B: `hybrid_hybrid_label_foundry_hybrid_hybrid_minimu_m542_s0.py` – provides
  Bayesian label‑confidence utilities, morphology indices and related helpers.

Mathematical Bridge:
The scalar (grade‑0) part of the geometric product of two vectors in a Clifford
algebra equals their Euclidean dot product.  We exploit this property to turn the
Clifford product into a *context‑sensitive similarity* between feature vectors.
That similarity then modulates the Bayesian update performed by Parent B,
yielding a hybrid confidence that is aware of geometric context while retaining
the probabilistic rigor of the original labeling framework.
"""

import sys
import math
import random
import pathlib
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Any, Tuple

# ----------------------------------------------------------------------
# Parent A – Clifford algebra core
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicate indices cancel (e_i ∧ e_i = 0) and contribute a sign flip for each
    swap needed to sort the list.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel the pair and restart scanning from the beginning
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                n -= 2
                i = -1  # will become 0 after i += 1 at loop bottom
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a: Dict[frozenset, float],
                     b: Dict[frozenset, float]) -> Dict[frozenset, float]:
    """Full Clifford product `ab`.

    Parameters
    ----------
    a, b : dict
        Mapping from basis blade (frozenset of integer indices) to scalar coefficient.

    Returns
    -------
    dict
        Multivector representing the product.
    """
    result: Dict[frozenset, float] = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade, sign = _multiply_blades(blade_a, blade_b)
            result[blade] = result.get(blade, 0.0) + sign * coef_a * coef_b
    return result

# ----------------------------------------------------------------------
# Helper: convert a real‑valued vector to a multivector (only grade‑1 blades)
# ----------------------------------------------------------------------
def multivector_from_vector(vec: np.ndarray) -> Dict[frozenset, float]:
    """Map a 1‑D numpy array to a Clifford multivector.

    Each component `vec[i]` becomes the coefficient of the basis blade `e_i`
    (represented as ``frozenset({i})``). Zero components are omitted.
    """
    mv: Dict[frozenset, float] = {}
    for i, coeff in enumerate(vec):
        if coeff != 0.0:
            mv[frozenset({i})] = float(coeff)
    return mv

# ----------------------------------------------------------------------
# Parent B – Bayesian labeling core (trimmed to essentials)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability P(E)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior P(H|E)."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def clifford_scalar_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similarity based on the scalar part of the geometric product.

    The scalar part equals the Euclidean dot product.  The function returns a
    cosine‑like similarity in the range [0, 1] (negative dot products are clamped
    to 0 to keep the Bayesian interpretation valid).
    """
    if v1.shape != v2.shape:
        raise ValueError("vectors must have the same shape")
    mv1 = multivector_from_vector(v1)
    mv2 = multivector_from_vector(v2)
    product = geometric_product(mv1, mv2)
    # The scalar blade is represented by the empty frozenset
    dot = product.get(frozenset(), 0.0)
    norm1 = math.sqrt(sum(float(c) ** 2 for c in v1))
    norm2 = math.sqrt(sum(float(c) ** 2 for c in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    cosine = dot / (norm1 * norm2)
    return max(0.0, min(1.0, cosine))

def hybrid_bayes_confidence(prior: float,
                            v_observed: np.ndarray,
                            v_reference: np.ndarray,
                            likelihood: float) -> float:
    """Compute a Bayesian posterior where the similarity modulates the likelihood.

    Parameters
    ----------
    prior : float
        Prior probability of the hypothesis.
    v_observed, v_reference : np.ndarray
        Feature vectors describing the observed document and a reference prototype.
    likelihood : float
        Base likelihood P(E|H) before similarity weighting.

    Returns
    -------
    float
        Posterior probability P(H|E) incorporating geometric similarity.
    """
    similarity = clifford_scalar_similarity(v_observed, v_reference)
    # Use similarity as a confidence weight on the likelihood.
    weighted_likelihood = likelihood * similarity
    # Treat (1 - similarity) as an effective false‑positive rate.
    false_positive = 1.0 - similarity
    marginal = bayes_marginal(prior, weighted_likelihood, false_positive)
    posterior = bayes_update(prior, weighted_likelihood, marginal)
    return posterior

def morphology_features(length: float, width: float, height: float, mass: float) -> Tuple[float, float, float]:
    """Return (sphericity, flatness, righting_time) indices for a given morphology."""
    if min(length, width, height, mass) <= 0.0:
        raise ValueError("All morphology dimensions must be positive")
    sph = (length * width * height) ** (1.0 / 3.0) / length
    flat = (length + width) / (2.0 * height)
    # Simple righting‑time model (parameters chosen for demonstration)
    b = 1.0 / 3.0
    k = 0.35
    neck_lever = 1.0
    righting = b * (length / neck_lever) + k * mass
    return sph, flat, righting

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic feature vectors (e.g., TF‑IDF or embedding coordinates)
    vec_a = np.array([0.3, 0.5, 0.0, 0.2])
    vec_b = np.array([0.4, 0.4, 0.1, 0.1])

    sim = clifford_scalar_similarity(vec_a, vec_b)
    print(f"Clifford‑based similarity: {sim:.4f}")

    prior_prob = 0.6
    base_likelihood = 0.85
    posterior = hybrid_bayes_confidence(prior_prob, vec_a, vec_b, base_likelihood)
    print(f"Hybrid Bayesian posterior: {posterior:.4f}")

    # Morphology demo
    sph, flat, rt = morphology_features(length=2.0, width=1.5, height=1.0, mass=0.8)
    print(f"Sphericity: {sph:.4f}, Flatness: {flat:.4f}, Righting time: {rt:.4f}")