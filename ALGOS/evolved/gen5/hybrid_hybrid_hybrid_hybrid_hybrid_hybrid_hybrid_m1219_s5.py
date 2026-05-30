# DARWIN HAMMER — match 1219, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s1.py (gen4)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm: Stylometry‑TTT Fusion with Bayesian Cost and Differential Privacy

Parents:
- hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (stylometry features & Bayesian VRAM budgeting)
- hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s1.py (TTT‑Linear weight matrix, SSIM similarity, Laplace DP)

Mathematical Bridge:
The stylometry feature vector `f ∈ ℝⁿ` (extracted from text) is fed as the input `x` to the
TTT‑Linear transformation `W ∈ ℝ^{m×n}` producing `y = W x`.  
The transformed representation `y` is interpreted as a *likelihood* term in a Bayesian
cost model `C = -log π + λ·‖y‖₁` where `π` is a prior over stylistic classes and `λ`
scales the VRAM‑budget penalty.  
To assess fidelity we compute the Structural Similarity Index (SSIM) between the
original feature vector `f` and a normalized version of `y`.  
Finally a Laplace mechanism injects noise `η ~ Lap(ΔC/ε)` into the cost, yielding a
differentially‑private hybrid score.

The module implements this pipeline with three core functions:
`stylometry_vector`, `ttt_transform`, and `hybrid_cost`.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Stylometry utilities (excerpted from Parent A)
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _clean_word(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalpha())


def words(text: str) -> List[str]:
    """Tokenize `text` into lower‑case alphabetic words."""
    return [_clean_word(w) for w in (text or "").split() if _clean_word(w)]


def stylometry_vector(text: str) -> np.ndarray:
    """
    Build a normalized stylometry feature vector.

    The vector concatenates, in a fixed order, the relative frequencies of each
    FUNCTION_CATS category and the overall word‑length distribution (bins 1‑10+).
    The result is a 1‑D NumPy array of shape (n_features,).
    """
    ws = words(text)
    total = max(1, len(ws))

    # Category frequencies
    cat_vals = []
    for cat, vocab in FUNCTION_CATS.items():
        count = sum(1 for w in ws if w in vocab)
        cat_vals.append(count / total)

    # Word‑length histogram (bins 1‑10, 10+)
    length_counts = [0] * 11
    for w in ws:
        l = len(w)
        idx = l if l < 10 else 10
        length_counts[idx] += 1
    length_hist = [c / total for c in length_counts]

    feature_vec = np.array(cat_vals + length_hist, dtype=float)
    # Ensure unit‑norm for later SSIM comparison
    norm = np.linalg.norm(feature_vec) + 1e-12
    return feature_vec / norm


# ----------------------------------------------------------------------
# TTT‑Linear utilities (excerpted from Parent B)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a random TTT weight matrix W ∈ ℝ^{d_out×d_in}."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_transform(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear TTT transformation y = W @ x."""
    return W @ x


# ----------------------------------------------------------------------
# Bayesian cost integration + differential privacy (concept from Parent A)
# ----------------------------------------------------------------------
def bayesian_cost(
    transformed: np.ndarray,
    prior: Dict[int, float] | None = None,
    lambda_budget: float = 1.0,
) -> float:
    """
    Compute a Bayesian‑style cost:
        C = -log π_k + λ * ||y||_1
    where k = argmax_i |y_i| (the dominant stylistic class) and π_k is its prior.
    """
    if prior is None:
        # Uniform prior over the dimension of `transformed`
        dim = transformed.shape[0]
        prior = {i: 1.0 / dim for i in range(dim)}

    # Identify dominant class index
    k = int(np.argmax(np.abs(transformed)))
    pi_k = prior.get(k, 1e-6)  # avoid log(0)

    l1_norm = np.sum(np.abs(transformed))
    cost = -math.log(pi_k) + lambda_budget * l1_norm
    return cost


def laplace_mechanism(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    """
    Apply Laplace noise to `value` with privacy budget ε.
    The global ℓ1‑sensitivity is supplied (default 1.0).
    """
    scale = sensitivity / max(epsilon, 1e-12)
    noise = np.random.laplace(0.0, scale)
    return value + noise


# ----------------------------------------------------------------------
# SSIM similarity (from Parent B)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 1.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Compute a simplified SSIM index for 1‑D signals.
    Both inputs are assumed to be normalized to [0, dynamic_range].
    """
    if x.shape != y.shape:
        raise ValueError("inputs must have identical shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(numerator / (denominator + 1e-12))


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_score(
    text: str,
    W: np.ndarray,
    prior: Dict[int, float] | None = None,
    lambda_budget: float = 1.0,
    epsilon: float = 0.5,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid computation:

    1. Extract stylometry vector `f`.
    2. Transform via TTT matrix → `y`.
    3. Compute Bayesian cost `C`.
    4. Add Laplace noise → `Ĉ`.
    5. Evaluate SSIM between `f` and a normalized version of `y`.

    Returns a dictionary with intermediate results and the final private score.
    """
    # 1. Stylometry
    f = stylometry_vector(text)

    # 2. TTT transformation (ensure compatible dimensions)
    if W.shape[1] != f.shape[0]:
        raise ValueError(f"Weight matrix column size {W.shape[1]} does not match feature size {f.shape[0]}")
    y_raw = ttt_transform(W, f)

    # Normalise transformed vector to [0, 1] for SSIM
    y_min, y_max = y_raw.min(), y_raw.max()
    if y_max - y_min < 1e-12:
        y_norm = np.zeros_like(y_raw)
    else:
        y_norm = (y_raw - y_min) / (y_max - y_min)

    # 3. Bayesian cost
    cost = bayesian_cost(y_raw, prior, lambda_budget)

    # 4. Differential privacy
    private_cost = laplace_mechanism(cost, epsilon, sensitivity=1.0)

    # 5. SSIM similarity
    similarity = ssim(f, y_norm, dynamic_range=1.0)

    return {
        "stylometry_vector": f,
        "ttt_output_raw": y_raw,
        "ttt_output_norm": y_norm,
        "bayesian_cost": cost,
        "private_cost": private_cost,
        "ssim_similarity": similarity,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "In the midst of the quiet forest, the wind whispered secrets to the ancient trees, "
        "while birds sang melodies that echoed across the canopy."
    )

    # Initialise a weight matrix compatible with the feature dimension
    feat_dim = stylometry_vector(sample_text).shape[0]
    W = init_ttt(d_in=feat_dim, d_out=feat_dim, scale=0.05, seed=42)

    # Uniform prior (implicit in function)
    result = hybrid_score(
        text=sample_text,
        W=W,
        prior=None,
        lambda_budget=0.8,
        epsilon=0.7,
    )

    # Print a concise summary
    print("Hybrid Score Summary")
    print("--------------------")
    print(f"Bayesian cost (raw): {result['bayesian_cost']:.4f}")
    print(f"Private cost (ε={0.7}): {result['private_cost']:.4f}")
    print(f"SSIM similarity: {result['ssim_similarity']:.4f}")
    # Ensure the script exits without error
    sys.exit(0)