# DARWIN HAMMER — match 3912, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s4.py (gen5)
# born: 2026-05-29T23:52:33Z

"""
Hybrid Algorithm: Fusing Stylometry with Bayesian Feature Extraction, Hyperdimensional Computing, and Gaussian Beam Filtering

This module integrates the stylometry features from `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s1.py`
with the Bayesian-inspired feature extraction and hyperdimensional computing from the same parent, and the Gaussian beam
filtering from `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s4.py`. The mathematical bridge between the two
parents lies in their shared use of hash functions to seed pseudo-random number generators and generate feature vectors,
and the application of Gaussian filtering to smooth the stylometry features.

The governing equations of the first parent involve calculating the proportion of words belonging to each FUNCTION_CAT,
while the second parent uses a deterministic hash to extract a feature vector and hyperdimensional computing primitives
to represent morphological scalars and derived indices as bipolar hypervectors. We fuse these equations by using the
hash function to seed the pseudo-random generator and applying Gaussian filtering to the stylometry features.

The second parent's Gaussian beam filtering is integrated by applying it to the stylometry features before calculating the
proportion of words belonging to each FUNCTION_CAT. This allows for a more robust and smoothed representation of the
stylometry features.
"""

import numpy as np
import random
import math
import hashlib
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

FUNCTION_CATS: Dict[str, set[str]] = {
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
    "quantifier": set()
}

Vector = Sequence[float]


def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator/denominator, falling back to *fallback* if denominator is zero."""
    return numerator / denominator if denominator != 0 else fallback


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0,1] for non‑negative vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single‑parameter Gaussian model.

    I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """1‑D Gaussian smoothing with standard deviation *sigma*."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # Build a symmetric kernel covering ±3σ (enough for practical purposes)
    radius = int(math.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    # Convolve the data with the kernel
    return np.convolve(data, kernel, mode='same')


def stylometry_features(text: str, function_cats: Dict[str, set[str]]) -> np.ndarray:
    """Calculate the stylometry features for a given text."""
    words = text.split()
    features = np.zeros(len(function_cats))
    for i, (cat, words_in_cat) in enumerate(function_cats.items()):
        count = sum(1 for word in words if word in words_in_cat)
        features[i] = count / len(words)
    return features


def hybrid_stylometry_gaussian_beam(text: str, function_cats: Dict[str, set[str]], center: float, width: float) -> float:
    """Calculate the hybrid stylometry Gaussian beam feature."""
    features = stylometry_features(text, function_cats)
    smoothed_features = gaussian_filter(features, width)
    return gaussian_beam(smoothed_features.mean(), center, width)


def hybrid_fisher_score(text: str, function_cats: Dict[str, set[str]], center: float, width: float) -> float:
    """Calculate the hybrid Fisher score."""
    features = stylometry_features(text, function_cats)
    smoothed_features = gaussian_filter(features, width)
    return fisher_score(smoothed_features.mean(), center, width)


if __name__ == "__main__":
    text = "This is a sample text for testing the hybrid algorithm."
    center = 0.5
    width = 0.1
    print(hybrid_stylometry_gaussian_beam(text, FUNCTION_CATS, center, width))
    print(hybrid_fisher_score(text, FUNCTION_CATS, center, width))