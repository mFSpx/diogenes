# DARWIN HAMMER — match 3149, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s1.py (gen5)
# born: 2026-05-29T23:48:07Z

"""Hybrid Algorithm: Fusion of SSIM‑Based Signal Similarity (Parent A) with
Stylometry‑Driven Bayesian Cost and Ternary Routing (Parent B)

Mathematical Bridge
-------------------
Parent A provides a structural similarity index (SSIM) computed on two
signal matrices together with an entropy term and a VRAM‑budget scaling
factor.  
Parent B supplies a feature vector **f** derived from stylometric
categories, a Bayesian tree‑cost  C = −∑ fᵢ log πᵢ (π being a prior
distribution) and a ternary router that decides how a record should be
handled.

The fusion treats the stylometric feature vector as a *prior* over the
signal space: the SSIM numerator (2μₓμᵧ + C₁) and denominator
(μₓ² + μᵧ² + C₁) are weighted by exp(−β C) where β controls the influence
of the Bayesian cost.  Entropy **H** of the signal and the VRAM scaling
factor **γ** further modulate the final hybrid similarity score **S**:


γ   = min(1,  budget_MB / usage_MB)
C   = -∑_i f_i * log(π_i + ε)
S   = γ * (SSIM(signal_a, signal_b) * e^{-β C}) * (1 - H / H_max)


The resulting score drives the ternary router (store / defer / evict).

This module implements the combined mathematics in a self‑contained,
executable form.
"""

import os
import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Stylometry utilities (Parent B)
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _tokenize(text: str) -> List[str]:
    """Very light tokenizer that removes punctuation and lower‑cases."""
    for ch in PUNCT:
        text = text.replace(ch, " ")
    return [t for t in text.lower().split() if t]


def stylometry_features(text: str) -> np.ndarray:
    """
    Returns a normalized feature vector (length = number of categories)
    where each entry is the proportion of words belonging to the category.
    """
    tokens = _tokenize(text or "")
    total = len(tokens) or 1
    feats = []
    for cat, vocab in FUNCTION_CATS.items():
        count = sum(1 for w in tokens if w in vocab)
        feats.append(count / total)
    return np.array(feats, dtype=np.float64)


def uniform_prior(num_features: int) -> np.ndarray:
    """Uniform prior π_i = 1 / N for all features."""
    return np.full(num_features, 1.0 / num_features, dtype=np.float64)


def bayesian_tree_cost(features: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> float:
    """
    Bayesian tree cost C = -∑ f_i * log(π_i)
    Small epsilon avoids log(0).
    """
    return -float(np.sum(features * np.log(prior + eps)))


# ----------------------------------------------------------------------
# Signal generation & SSIM utilities (Parent A)
# ----------------------------------------------------------------------
def _signal_from_features(features: np.ndarray, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Create a synthetic 2‑D signal matrix whose rows are scaled copies of the
    feature vector.  This provides a deterministic bridge from stylometry to
    a numeric signal domain.
    """
    # Broadcast the feature vector to the requested shape
    vec = features.reshape(1, -1)
    repeats = (size[0], max(1, size[1] // vec.shape[1]))
    signal = np.tile(vec, repeats)
    # Clip to [0, 255] and cast to uint8 for SSIM compatibility
    signal = np.clip(signal * 255.0, 0, 255).astype(np.uint8)
    # Pad or truncate to exact size
    if signal.shape != size:
        signal = np.resize(signal, size)
    return signal


def ssim(img1: np.ndarray, img2: np.ndarray, K1: float = 0.01, K2: float = 0.03, L: int = 255) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two 2‑D arrays.
    Implements the classic formula without external dependencies.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / (denominator + 1e-12))


def shannon_entropy(matrix: np.ndarray, bins: int = 256) -> float:
    """
    Compute Shannon entropy (base‑2) of the intensity distribution of a matrix.
    """
    hist, _ = np.histogram(matrix.ravel(), bins=bins, range=(0, 255), density=True)
    p = hist + 1e-12  # avoid log(0)
    return float(-np.sum(p * np.log2(p)))


def vram_scaling(current_mb: float, budget_mb: float) -> float:
    """
    Scaling factor γ = min(1, budget / current).  If current usage exceeds the
    budget the factor reduces the final score proportionally.
    """
    if current_mb <= 0:
        return 1.0
    return min(1.0, budget_mb / current_mb)


# ----------------------------------------------------------------------
# Hybrid operation (fusion of A & B)
# ----------------------------------------------------------------------
def hybrid_similarity(
    text_a: str,
    text_b: str,
    current_vram_mb: float,
    budget_vram_mb: float,
    beta: float = 0.5,
) -> float:
    """
    Compute the fused similarity score S between two texts.

    Steps:
    1. Extract stylometry feature vectors f_a, f_b.
    2. Build uniform prior π.
    3. Compute Bayesian costs C_a, C_b.
    4. Generate synthetic signals s_a, s_b from the feature vectors.
    5. Compute SSIM(s_a, s_b) and entropy H of the average signal.
    6. Apply VRAM scaling γ.
    7. Combine everything according to the bridge equation.
    """
    # 1. Stylometry
    f_a = stylometry_features(text_a)
    f_b = stylometry_features(text_b)

    # 2. Prior (same for both texts)
    prior = uniform_prior(len(f_a))

    # 3. Bayesian costs
    C_a = bayesian_tree_cost(f_a, prior)
    C_b = bayesian_tree_cost(f_b, prior)
    C = (C_a + C_b) / 2.0

    # 4. Synthetic signals
    sig_a = _signal_from_features(f_a)
    sig_b = _signal_from_features(f_b)

    # 5. SSIM and entropy
    ssim_val = ssim(sig_a, sig_b)
    avg_sig = ((sig_a.astype(np.float64) + sig_b.astype(np.float64)) / 2.0).astype(np.uint8)
    H = shannon_entropy(avg_sig)
    H_max = math.log2(256)  # max entropy for 8‑bit intensity

    # 6. VRAM scaling
    gamma = vram_scaling(current_vram_mb, budget_vram_mb)

    # 7. Fusion equation
    score = (
        gamma
        * ssim_val
        * math.exp(-beta * C)
        * (1.0 - H / H_max)
    )
    return float(score)


def ternary_router(score: float, low_thr: float = 0.3, high_thr: float = 0.7) -> str:
    """
    Decide what to do with a record based on the hybrid score.

    - score < low_thr   → 'evict'
    - score > high_thr  → 'store'
    - otherwise         → 'defer'
    """
    if score < low_thr:
        return "evict"
    if score > high_thr:
        return "store"
    return "defer"


def batch_hybrid_process(
    texts: List[Tuple[str, str]],
    current_vram_mb: float,
    budget_vram_mb: float,
) -> List[Tuple[float, str]]:
    """
    Process a batch of text pairs, returning a list of (score, action) tuples.
    Demonstrates the hybrid pipeline on multiple inputs.
    """
    results = []
    for a, b in texts:
        sc = hybrid_similarity(a, b, current_vram_mb, budget_vram_mb)
        act = ternary_router(sc)
        results.append((sc, act))
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_pairs = [
        ("I love programming in Python.", "I enjoy coding with Python."),
        ("The quick brown fox jumps over the lazy dog.", "Pack my box with five dozen liquor jugs."),
        ("Data science is fun.", "Quantum mechanics is bewildering."),
    ]
    # Simulate a VRAM usage scenario
    current_usage = 1024.0  # MB
    budget = 2048.0         # MB

    print("Hybrid similarity scores and actions:")
    for idx, (txt1, txt2) in enumerate(sample_pairs, 1):
        score = hybrid_similarity(txt1, txt2, current_usage, budget)
        action = ternary_router(score)
        print(f"Pair {idx}: score={score:.4f} → {action}")

    # Batch demonstration
    batch = batch_hybrid_process(sample_pairs, current_usage, budget)
    print("\nBatch results:")
    for i, (sc, act) in enumerate(batch, 1):
        print(f"{i}. score={sc:.4f}, action={act}")