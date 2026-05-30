# DARWIN HAMMER — match 5148, survivor 2
# gen: 6
# parent_a: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_rlct_grokking_m1358_s0.py (gen5)
# born: 2026-05-30T00:00:16Z

"""Hybrid Algorithm: Serpentina Morphology ↔ Stylometry RLCT Fusion

Parent A: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s0.py  
Parent B: hybrid_hybrid_hybrid_hard_t_hybrid_rlct_grokking_m1358_s0.py  

Mathematical Bridge
-------------------
The bridge is built on two scalar quantities that both algorithms treat as
modulators of a core information measure:

* **Sphericity index** (from the serpentina self‑righting morphology) quantifies
  how close the turtle’s shape is to a sphere. It is used in Parent A to scale
  the Fisher information of a Gaussian beam.

* **Real Log Canonical Threshold (RLCT)** (from the Bayesian stylometry pipeline)
  regularizes the posterior feature distribution.

In the hybrid we let the sphericity index act as a *scale* for the RLCT
regularization and let the resulting regularized posterior be *weighted* by the
Fisher information computed for the same morphology.  The final scalar metric
is therefore


HybridScore = FisherScore(sphericity) × Σ  RLCTRegularize( BayesUpdate(prior, LSM(text)) , RLCT )


Thus morphological information directly influences the statistical
decision‑making pipeline.

The module implements the combined workflow with clear, reusable functions.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Fisher information
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Mean radius divided by the longest dimension (length)."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Simple flatness measure used in the original righting time formula."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length + width) / (2.0 * height)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float,
                 center: float,
                 width: float,
                 sphericity: float = 1.0,
                 eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam, scaled by sphericity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return sphericity * (derivative * derivative) / intensity


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0,
                        theta: float = 0.0,
                        center: float = 0.0,
                        width: float = 1.0) -> float:
    """
    Simplified version of the original right‑handed time index.
    It combines flatness, sphericity and Fisher information.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    fisher = fisher_score(theta, center, width, sphericity=sph)

    # Core formula (invented for the hybrid): mass‑scaled flatness × Fisher
    return (m.mass ** b) * (flat ** k) * fisher * neck_lever


# ----------------------------------------------------------------------
# Parent B – Stylometry, Bayesian update & RLCT regularization
# ----------------------------------------------------------------------


FUNCTION_CATS: Dict[str, set] = {
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
}


@dataclass
class StylometryFeatures:
    cat: str
    vocab: List[str]
    cnt: List[int]
    total: int


def words(text: str) -> List[str]:
    """Tokenize simple alphabetic words, lower‑cased."""
    return [w.lower() for w in text.split() if w.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a lightweight stylistic matrix (LSM) vector:
    frequency of each functional category normalized by total token count.
    """
    toks = words(text)
    if not toks:
        raise ValueError("Input text must contain at least one alphabetic token")
    counts = {cat: sum(1 for w in toks if w in FUNCTION_CATS[cat]) for cat in FUNCTION_CATS}
    return {cat: counts[cat] / len(toks) for cat in FUNCTION_CATS}


def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Simple element‑wise Bayesian update (unnormalized).
    posterior ∝ prior × likelihood.
    """
    if prior.shape != likelihood.shape:
        raise ValueError("Prior and likelihood must have identical shape")
    posterior = prior * likelihood
    norm = posterior.sum()
    return posterior / norm if norm > 0 else posterior


def estimate_rlct_from_losses(losses: List[float],
                              n_params: int,
                              n_samples: int) -> float:
    """
    Very coarse estimator of the Real Log Canonical Threshold (RLCT).

    The original RLCT relates to the curvature of the loss landscape.
    Here we approximate it by the ratio of variance to mean scaled by the
    model size.
    """
    if not losses:
        raise ValueError("Loss list cannot be empty")
    losses_arr = np.asarray(losses, dtype=float)
    mean = losses_arr.mean()
    var = losses_arr.var()
    if mean == 0:
        mean = 1e-12
    rlct = (var / mean) * (n_params / max(1, n_samples))
    return max(rlct, 1e-12)


def rlct_regularization(posterior: np.ndarray, rlct: float) -> np.ndarray:
    """
    Apply RLCT regularization by attenuating the posterior with an
    exponential decay governed by the RLCT value.
    """
    factor = np.exp(-rlct * np.abs(posterior))
    return posterior * factor


# ----------------------------------------------------------------------
# Hybrid Functions (demonstrating the fused system)
# ----------------------------------------------------------------------


def compute_morphology_fisher(m: Morphology,
                              theta: float = 0.0,
                              center: float = 0.0,
                              width: float = 1.0) -> float:
    """
    Wrapper that returns the Fisher information scaled by the morphology's
    sphericity index.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    return fisher_score(theta, center, width, sphericity=sph)


def stylometry_posterior(text: str) -> np.ndarray:
    """
    Produce a normalized Bayesian posterior over functional categories
    based on the LSM vector of the supplied text.
    """
    vec_dict = lsm_vector(text)
    v = np.array([vec_dict[cat] for cat in sorted(vec_dict.keys())])
    prior = np.ones_like(v) / v.size
    return bayes_update(prior, v)


def hybrid_score(m: Morphology,
                 text: str,
                 losses: List[float],
                 theta: float = 0.0,
                 center: float = 0.0,
                 width: float = 1.0) -> float:
    """
    Core fused metric.

    1. Compute Fisher information using morphology (Parent A).
    2. Obtain a Bayesian posterior from stylometry (Parent B).
    3. Estimate RLCT from a list of losses and regularize the posterior.
    4. Weight the summed regularized posterior by the Fisher score.

    Returns a single scalar that can be interpreted as a "morpho‑stylometric
    confidence" measure.
    """
    # Step 1 – Fisher information modulated by sphericity
    fisher = compute_morphology_fisher(m, theta, center, width)

    # Step 2 – Bayesian posterior over functional categories
    posterior = stylometry_posterior(text)

    # Step 3 – RLCT regularization
    rlct = estimate_rlct_from_losses(
        losses,
        n_params=posterior.size,
        n_samples=len(losses)
    )
    regularized = rlct_regularization(posterior, rlct)

    # Step 4 – Final hybrid score
    return fisher * float(regularized.sum())


def combined_righting_and_stylometry(m: Morphology,
                                     text: str,
                                     losses: List[float]) -> Dict[str, float]:
    """
    Produce a dictionary with two related hybrid outputs:
    * ``righting_time`` – the morphology‑based righting time index (Parent A).
    * ``hybrid_score`` – the fused morpho‑stylometric confidence (see
      ``hybrid_score``).
    """
    rt = righting_time_index(m)
    hs = hybrid_score(m, text, losses)
    return {"righting_time": rt, "hybrid_score": hs}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample morphology resembling a small turtle
    morph = Morphology(length=0.30, width=0.20, height=0.10, mass=0.5)

    # Example text fragment
    sample_text = (
        "I think that the quick brown fox jumps over the lazy dog while "
        "you watch the sunrise. It is a beautiful day."
    )

    # Synthetic loss history (e.g., from a learning curve)
    synthetic_losses = [random.uniform(0.1, 1.0) for _ in range(50)]

    results = combined_righting_and_stylometry(morph, sample_text, synthetic_losses)

    print("Hybrid computation results:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")