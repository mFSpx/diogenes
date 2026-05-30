# DARWIN HAMMER — match 5247, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_krampu_m787_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s4.py (gen5)
# born: 2026-05-30T00:01:02Z

"""Hybrid Algorithm integrating Morphological Indices (Parent A) with Stylometric Bayesian Scoring (Parent B).

Mathematical bridge:
- Parent A provides a 4‑dimensional morphological feature vector **M** (sphericity, flatness,
  righting‑time, recovery‑priority).
- Parent B supplies a categorical stylometry count vector **S** derived from FUNCTION_CATS.
- The bridge is a *ternary routing matrix* **R** ∈ {‑1,0,1}^{4×C} (C = number of stylometry categories) that
  linearly mixes the two vectors: **X = R·S + M**.
- The mixed signal **X(t)** is then smoothed by a *Caputo fractional derivative* of order α,
  implemented via the Grünwald‑Letnikov series using the Lanczos approximation for Γ.
- Finally a Bayesian update combines a prior belief about “acceptability” with a likelihood
  derived from the fractional derivative magnitude, yielding a posterior score that can be
  released under differential privacy (Laplace noise).

The module implements this pipeline and provides three public functions demonstrating the
hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – Morphology utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def morphology_vector(m: Morphology) -> np.ndarray:
    """Return the 4‑dimensional feature vector [sphericity, flatness, righting, priority]."""
    sph = sphericity_index(m.length, m.width, m.height)
    flt = flatness_index(m.length, m.width, m.height)
    rgt = righting_time_index(m)
    rec = recovery_priority(m)
    return np.array([sph, flt, rgt, rec], dtype=float)


# ----------------------------------------------------------------------
# Parent B – Stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always".split()
    ),
}


def stylometry_vector(text: str) -> np.ndarray:
    """Count occurrences of each FUNCTION_CATS category in *text* and return a vector."""
    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    counts = []
    for cat in FUNCTION_CATS.values():
        cnt = sum(1 for t in tokens if t in cat)
        counts.append(cnt)
    return np.array(counts, dtype=float)


# ----------------------------------------------------------------------
# Shared mathematical tools
# ----------------------------------------------------------------------
def gamma_lanczos(z: float) -> float:
    """Lanczos approximation for the Gamma function (valid for z > 0)."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    # Shift to (z-1)
    z -= 1
    # Coefficients for g=7, n=9
    p = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ])
    g = 7
    a = p[0]
    for i in range(1, len(p)):
        a += p[i] / (z + i)
    t = z + g + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * a


def binomial_coeff(alpha: float, k: int) -> float:
    """Generalized binomial coefficient using Gamma."""
    return gamma_lanczos(alpha + 1) / (gamma_lanczos(k + 1) * gamma_lanczos(alpha - k + 1))


def caputo_fractional_derivative(signal: np.ndarray, order: float, dt: float = 1.0) -> np.ndarray:
    """
    Approximate the Caputo fractional derivative of *signal* of order *order*
    using the Grünwald‑Letnikov scheme.
    """
    n = len(signal)
    deriv = np.zeros_like(signal, dtype=float)
    for i in range(n):
        s = 0.0
        for k in range(i + 1):
            coeff = (-1) ** k * binomial_coeff(order, k)
            s += coeff * signal[i - k]
        deriv[i] = s / (dt ** order)
    return deriv


def ternary_router(rows: int, cols: int, seed: int | None = None) -> np.ndarray:
    """Generate a random ternary matrix with entries in {‑1,0,1}."""
    rng = random.Random(seed)
    data = [rng.choice([-1, 0, 1]) for _ in range(rows * cols)]
    return np.array(data, dtype=int).reshape(rows, cols)


def differential_privacy(value: float, epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Add Laplace noise to *value* for (ε, Δ)-differential privacy."""
    scale = sensitivity / epsilon
    noise = random.laplacevariate(0.0, scale) if hasattr(random, "laplacevariate") else np.random.laplace(0.0, scale)
    return value + float(noise)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_feature_mix(morph: Morphology, text: str,
                       router: np.ndarray) -> np.ndarray:
    """
    Combine morphological vector **M** with stylometry vector **S** via ternary router **R**.
    Returns X = M + R·S.
    """
    M = morphology_vector(morph)                 # shape (4,)
    S = stylometry_vector(text)                  # shape (C,)
    if router.shape[1] != S.shape[0]:
        raise ValueError("Router column count must match stylometry vector length")
    mixed = M + router @ S
    return mixed


def hybrid_score(morph: Morphology, text: str,
                 order: float = 0.5,
                 epsilon: float = 0.5,
                 seed: int | None = None) -> float:
    """
    Full pipeline:
    1. Mix features with a ternary router.
    2. Apply Caputo fractional derivative (order α).
    3. Compute a likelihood from the derivative magnitude.
    4. Perform a Bayesian update with a uniform prior (0.5).
    5. Release the posterior under differential privacy.
    Returns the privatized posterior probability that the input is “acceptable”.
    """
    # 1. Router (4 × C)
    router = ternary_router(4, len(FUNCTION_CATS), seed=seed)

    # 2. Mixed signal (temporal series of length 4)
    X = hybrid_feature_mix(morph, text, router)

    # 3. Fractional derivative of the mixed signal
    dX = caputo_fractional_derivative(X, order)

    # 4. Likelihood: larger magnitude → lower likelihood of acceptance
    mag = np.linalg.norm(dX)
    likelihood = math.exp(-mag)  # in (0,1]

    # 5. Bayesian update with prior 0.5
    prior = 0.5
    posterior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood))

    # 6. Differential privacy
    return differential_privacy(posterior, epsilon=epsilon)


def batch_hybrid_scores(morphs: List[Morphology],
                        texts: List[str],
                        order: float = 0.5,
                        epsilon: float = 0.5) -> List[float]:
    """
    Apply *hybrid_score* to parallel lists of morphologies and texts.
    Returns a list of privatized posterior scores.
    """
    if len(morphs) != len(texts):
        raise ValueError("Morphology and text lists must be of equal length")
    scores = []
    for i, (m, t) in enumerate(zip(morphs, texts)):
        scores.append(hybrid_score(m, t, order=order, epsilon=epsilon, seed=i))
    return scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple morphology instance
    demo_morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Example text
    demo_text = ("The quick brown fox jumps over the lazy dog. "
                 "It is not only fast but also clever, and it never gives up.")

    # Run single hybrid score
    score = hybrid_score(demo_morph, demo_text, order=0.6, epsilon=0.8, seed=42)
    print(f"Hybrid posterior (private): {score:.4f}")

    # Batch test
    morphs = [demo_morph, Morphology(1.5, 0.8, 0.6, 2.5)]
    texts = [demo_text,
             "She does not like the rain, but she enjoys the sunshine and the warm breeze."]
    batch = batch_hybrid_scores(morphs, texts, order=0.5, epsilon=1.0)
    print("Batch scores:", ["{:.4f}".format(s) for s in batch])