# DARWIN HAMMER — match 1219, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s1.py (gen4)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm: Stylometry‑TTT Fusion with Bayesian Cost and Differential Privacy.

Parents:
- hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (stylometry features, Bayesian tree cost, VRAM budgeting)
- hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s1.py (TTT‑Linear weight matrix, SSIM similarity, Laplace DP noise)

Mathematical bridge:
The stylometric frequency vector **f** (derived from FUNCTION_CATS) is treated as a
feature column vector **x**.  It is multiplied by a TTT‑Linear weight matrix **W**
(the same matrix used for Count‑Min sketch construction).  The transformed
vector **z = W·x** serves as sufficient statistics for a Gaussian Bayesian
tree‑cost model.  The posterior cost is evaluated as a negative‑log‑likelihood
using a prior (μ, Σ).  To preserve privacy, Laplace noise calibrated by ε is
added to the cost, and structural similarity (SSIM) between the original
frequency vector and its reconstruction **W⁺·z** (pseudo‑inverse) quantifies
reconstruction risk.

The module therefore fuses:
1. Stylometric extraction → numeric vector.
2. TTT‑Linear linear map → transformed representation.
3. Bayesian cost computation on the transformed space.
4. Differential‑private noise injection.
5. SSIM‑based similarity assessment.

All operations rely only on NumPy and the Python standard library.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities (trimmed)
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

def words(text: str) -> List[str]:
    """Tokenize to alphabetic lower‑case words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def build_vocab() -> List[str]:
    """Flatten FUNCTION_CATS into a deterministic vocabulary list."""
    vocab = sorted({w for cat in FUNCTION_CATS.values() for w in cat})
    return vocab

VOCAB: List[str] = build_vocab()
VOCAB_INDEX: Dict[str, int] = {w: i for i, w in enumerate(VOCAB)}

def stylometry_vector(text: str) -> np.ndarray:
    """
    Return a normalized frequency vector over VOCAB.
    Frequency = count / max(1, total_words).
    """
    ws = words(text)
    total = max(1, len(ws))
    vec = np.zeros(len(VOCAB), dtype=float)
    for w in ws:
        idx = VOCAB_INDEX.get(w)
        if idx is not None:
            vec[idx] += 1.0
    vec /= total
    return vec

# ----------------------------------------------------------------------
# Parent B – TTT‑Linear utilities (trimmed)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a TTT‑Linear weight matrix W ~ N(0, scale²)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_transform(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear transformation z = W·x."""
    return W @ x

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 1.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("inputs must have the same shape")
    if x.size == 0:
        raise ValueError("inputs must be non‑empty")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(numerator / denominator)

def laplace_noise(scale: float) -> float:
    """Sample zero‑mean Laplace noise with given scale."""
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_transform(text: str, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract stylometry vector x from text, apply TTT transform to obtain z.
    Returns (x, z).
    """
    x = stylometry_vector(text)
    if W.shape[1] != x.shape[0]:
        raise ValueError("Weight matrix column size must match vocab dimension")
    z = ttt_transform(W, x)
    return x, z

def bayesian_cost(z: np.ndarray,
                  prior_mean: np.ndarray | None = None,
                  prior_var: float = 1.0) -> float:
    """
    Compute negative log‑posterior under a Gaussian prior N(prior_mean, prior_var·I).
    If prior_mean is None, it defaults to zero vector.
    """
    if prior_mean is None:
        prior_mean = np.zeros_like(z)
    diff = z - prior_mean
    # Σ⁻¹ = (1/prior_var)·I
    cost = 0.5 * np.sum(diff ** 2) / prior_var
    return float(cost)

def hybrid_cost_with_privacy(text: str,
                             W: np.ndarray,
                             epsilon: float = 1.0,
                             prior_mean: np.ndarray | None = None,
                             prior_var: float = 1.0) -> Tuple[float, float]:
    """
    Full pipeline:
    1. Transform text → (x, z)
    2. Compute Bayesian cost on z
    3. Add Laplace noise calibrated by ε (scale = 1/ε)
    4. Compute SSIM between original x and reconstruction W⁺·z
    Returns (noisy_cost, similarity).
    """
    x, z = hybrid_transform(text, W)
    cost = bayesian_cost(z, prior_mean, prior_var)
    # Differential privacy: Laplace mechanism on scalar cost
    noise_scale = 1.0 / max(epsilon, 1e-9)
    noisy_cost = cost + laplace_noise(noise_scale)

    # Reconstruction using Moore‑Penrose pseudo‑inverse (acts like Count‑Min sketch decode)
    try:
        W_pinv = np.linalg.pinv(W)
        x_recon = W_pinv @ z
    except np.linalg.LinAlgError:
        x_recon = np.zeros_like(x)

    similarity = ssim(x, x_recon, dynamic_range=1.0)
    return noisy_cost, similarity

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I think therefore I am. The quick brown fox jumps over the lazy dog, "
        "but the dog does not seem to mind."
    )
    # Initialise a weight matrix compatible with VOCAB size
    W = init_ttt(d_in=len(VOCAB), d_out=64, scale=0.05, seed=42)

    # Run hybrid pipeline
    noisy_cost, sim = hybrid_cost_with_privacy(
        sample_text,
        W,
        epsilon=0.5,
        prior_mean=None,
        prior_var=0.1,
    )

    print(f"Hybrid noisy cost: {noisy_cost:.4f}")
    print(f"Reconstruction SSIM similarity: {sim:.4f}")
    # Ensure outputs are finite
    assert math.isfinite(noisy_cost), "Cost should be a finite number"
    assert 0.0 <= sim <= 1.0, "SSIM should be in [0,1]"
    print("Smoke test passed.")