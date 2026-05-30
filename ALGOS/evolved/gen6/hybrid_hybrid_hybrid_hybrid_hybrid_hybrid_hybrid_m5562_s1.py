# DARWIN HAMMER — match 5562, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m1870_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s5.py (gen5)
# born: 2026-05-30T00:03:03Z

"""Hybrid Algorithm: Morphology‑Weighted Stylometry‑TTT Fusion with Bayesian Cost and Differential Privacy

Parents:
- hybrid_hybrid_percyphon_hyb_hybrid_hybrid_bayes__m163_s0.py (provides morphological analysis,
  sphericity index and a text‑based feature vector)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s5.py (provides stylometry extraction,
  a TTT‑linear transformation, Bayesian cost model, SSIM fidelity metric and Laplace DP)

Mathematical Bridge:
The sphericity index `σ = sphericity_index(L,W,H)` derived from a `Morphology` object is used as a
scalar weight on the stylometry feature vector `f`.  The weighted vector `f̃ = σ·f` is then fed
as input `x` to the TTT linear map `W ∈ ℝ^{m×n}` producing `y = W f̃`.  The transformed
representation is interpreted as a likelihood term inside a Bayesian cost

    C = -log π_c + λ·‖y‖₁

where `π_c` is the prior probability of the predicted stylistic class `c`.  To guarantee
ε‑differential privacy we inject Laplace noise `η ~ Lap(ΔC/ε)` (ΔC is conservatively set to 1)
into the cost, yielding the private hybrid score `Ĉ = C + η`.  Finally a 1‑D Structural
Similarity Index (SSIM) between the original stylometry vector `f` and a normalized version of
`y` quantifies fidelity.

The module below implements this unified pipeline with three core functions:
`stylometry_vector`, `ttt_transform`, and `morphology_weighted_hybrid_cost`.  A small smoke
test demonstrates end‑to‑end execution."""


import math
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Return the sphericity index defined in Parent A."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def morphology_from_dimensions(
    length: float, width: float, height: float, mass: float
) -> Morphology:
    """Convenient factory for a Morphology instance."""
    return Morphology(length=length, width=width, height=height, mass=mass)


# ----------------------------------------------------------------------
# Parent B – Stylometry & TTT utilities
# ----------------------------------------------------------------------


# Simple lexical categories used for stylometry extraction.
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
}


def _tokenise(text: str) -> List[str]:
    """Very small tokeniser – split on whitespace and strip punctuation."""
    return [t.strip(".,;:!?\"'()[]{}").lower() for t in text.split() if t]


def stylometry_vector(text: str) -> np.ndarray:
    """
    Produce a stylometry feature vector `f ∈ ℝ^k` where each component counts the
    occurrences of a lexical category defined in ``FUNCTION_CATS``.
    """
    tokens = _tokenise(text)
    counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    for idx, cat in enumerate(FUNCTION_CATS):
        counts[idx] = sum(1 for t in tokens if t in FUNCTION_CATS[cat])
    # Normalise to unit‑L2 length to keep magnitudes comparable across texts.
    norm = np.linalg.norm(counts)
    return counts / norm if norm > 0 else counts


def ttt_transform(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Linear TTT transformation (Parent B).  Computes ``y = W·x``.
    """
    if W.shape[1] != x.shape[0]:
        raise ValueError(f"Incompatible shapes: W{W.shape} @ x{x.shape}")
    return W @ x


def ssim_1d(a: np.ndarray, b: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Simplified 1‑D Structural Similarity Index.
    Implements the classic formula on flattened vectors.
    """
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2, sigma_b2 = ((a - mu_a) ** 2).mean(), ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid core – integration of both parents
# ----------------------------------------------------------------------


def morphology_weighted_hybrid_cost(
    text: str,
    morphology: Morphology,
    W: np.ndarray,
    prior: Dict[int, float],
    lam: float,
    epsilon: float,
    *,
    delta: float = 1.0,
) -> Tuple[float, float, int]:
    """
    End‑to‑end hybrid pipeline.

    1. Extract stylometry vector ``f`` from ``text`` (Parent B).
    2. Compute sphericity index ``σ`` from ``morphology`` (Parent A) and weight the vector:
           ``f̃ = σ·f``.
    3. Apply the TTT linear map: ``y = W·f̃``.
    4. Determine the most likely class ``c = argmax_i π_i`` using the supplied prior.
    5. Compute the Bayesian cost ``C = -log π_c + λ·‖y‖₁``.
    6. Add Laplace noise with scale ``ΔC/ε`` (ΔC is conservatively 1) to obtain a
       differentially‑private cost ``Ĉ``.
    7. Compute SSIM between the original ``f`` and a normalised version of ``y`` to
       measure fidelity.

    Returns:
        (private_cost, ssim, predicted_class)
    """
    # 1. Stylometry extraction
    f = stylometry_vector(text)  # shape (k,)

    # 2. Morphology weighting
    sigma = sphericity_index(morphology.length, morphology.width, morphology.height)
    f_weighted = sigma * f

    # 3. TTT transformation
    y = ttt_transform(W, f_weighted)  # shape (m,)

    # 4. Prior‑based class prediction (simple argmax)
    if not prior:
        raise ValueError("Prior dictionary must contain at least one class.")
    pred_class = max(prior, key=prior.get)
    pi_c = prior[pred_class]

    # 5. Bayesian cost
    cost = -math.log(pi_c) + lam * np.linalg.norm(y, ord=1)

    # 6. Differential privacy via Laplace mechanism
    scale = delta / epsilon if epsilon > 0 else float("inf")
    noise = np.random.laplace(loc=0.0, scale=scale)
    private_cost = cost + noise

    # 7. Fidelity metric (SSIM)
    # Normalise y to unit‑L2 to make it comparable with f.
    y_norm = y / np.linalg.norm(y) if np.linalg.norm(y) > 0 else y
    ssim_val = ssim_1d(f, y_norm)

    return private_cost, ssim_val, pred_class


def generate_random_weight_matrix(out_dim: int, in_dim: int, seed: int | None = None) -> np.ndarray:
    """
    Helper to create a reproducible random matrix ``W`` for the TTT transform.
    Entries are drawn from a standard normal distribution.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(out_dim, in_dim))


def example_prior(num_classes: int, seed: int | None = None) -> Dict[int, float]:
    """
    Generate a simple Dirichlet‑distributed prior over ``num_classes``.
    """
    rng = np.random.default_rng(seed)
    alpha = np.ones(num_classes)
    probs = rng.dirichlet(alpha)
    return {i: float(p) for i, p in enumerate(probs)}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Sample text
    sample_text = (
        "I cannot believe that you are not attending the meeting, "
        "but nevertheless we shall proceed without you."
    )

    # Morphology instance (arbitrary positive dimensions)
    morph = morphology_from_dimensions(length=2.3, width=1.8, height=0.9, mass=4.2)

    # Random weight matrix: output dimension 8, input dimension = len(FUNCTION_CATS)
    W = generate_random_weight_matrix(out_dim=8, in_dim=len(FUNCTION_CATS), seed=42)

    # Prior over 3 stylistic classes
    prior_dict = example_prior(num_classes=3, seed=7)

    # Hyper‑parameters
    lambda_penalty = 0.05
    epsilon_dp = 1.0  # reasonable privacy budget

    # Run hybrid pipeline
    private_cost, ssim_score, pred_cls = morphology_weighted_hybrid_cost(
        text=sample_text,
        morphology=morph,
        W=W,
        prior=prior_dict,
        lam=lambda_penalty,
        epsilon=epsilon_dp,
    )

    print(f"Private hybrid cost: {private_cost:.4f}")
    print(f"SSIM fidelity:        {ssim_score:.4f}")
    print(f"Predicted class:      {pred_cls}")
    print("All steps completed without error.")