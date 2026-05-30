# DARWIN HAMMER — match 3254, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m552_s1.py (gen5)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s1.py (gen2)
# born: 2026-05-29T23:50:15Z

import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import numpy as np

Vector = Sequence[float]

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def stylometry_analysis(text: str) -> Dict[str, int]:
    words = text.split()
    function_cats = {
        "pronoun": {
            "i", "me", "my", "mine", "myself", "you", "your", "yours",
            "yourself", "he", "him", "his", "she", "her", "hers",
            "they", "them", "their", "theirs", "we", "us", "our", "ours"
        },
        "article": {"a", "an", "the"},
        "preposition": {
            "about", "above", "after", "against", "around", "as", "at",
            "before", "behind", "below", "between", "by", "during", "for",
            "from", "in", "into", "of", "off", "on", "onto", "over",
            "through", "to", "under", "with", "without"
        },
        "auxiliary": {
            "am", "are", "be", "been", "being", "can", "could", "did",
            "do", "does", "had", "has", "have", "is", "may", "might",
            "must", "shall", "should", "was", "were", "will", "would"
        },
    }
    result = {cat: 0 for cat in function_cats}
    for word in words:
        lw = word.lower().strip(".,;:!?\"'")
        for cat, vocab in function_cats.items():
            if lw in vocab:
                result[cat] += 1
    return result

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")

def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))

def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        return 0.0
    if any(x < 0 for x in xs):
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = 0.0
    for i, x in enumerate(xs, start=1):
        cumulative += i * x
    total = sum(xs)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini

def fractional_power(x: float, alpha: float) -> float:
    if x == 0.0:
        return 0.0
    return math.copysign(abs(x) ** alpha, x)

def rbf_kernel(a: np.ndarray, b: np.ndarray, epsilon: float = 1.0) -> float:
    diff = a - b
    return math.exp(-epsilon * np.dot(diff, diff))

def rbf_surrogate_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    query: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    if train_X.shape[0] != train_y.shape[0]:
        raise ValueError("train_X and train_y must have same length")
    kernels = np.array([rbf_kernel(query, xi, epsilon) for xi in train_X])
    if kernels.sum() == 0:
        return float(np.mean(train_y))
    weights = kernels / kernels.sum()
    return float(np.dot(weights, train_y))

def hybrid_representation(data: List[float], text: str, dim: int = 4096) -> np.ndarray:
    phash = compute_phash(data)
    bits = [(phash >> i) & 1 for i in reversed(range(64))]
    phash_vec = np.array([1.0 if b else -1.0 for b in bits], dtype=np.complex128)
    repeats = dim // len(phash_vec) + 1
    phash_vec = np.tile(phash_vec, repeats)[:dim]

    hv = random_hv(d=dim, kind="complex", seed=42)

    bound = bind(hv, phash_vec)

    gini = gini_coefficient(data)
    alpha = 0.5 + 0.5 * gini 

    transformed = np.array([fractional_power(x, alpha) for x in data], dtype=np.float64)

    style_counts = stylometry_analysis(text)
    style_vec = np.array(
        [style_counts["pronoun"], style_counts["article"],
         style_counts["preposition"], style_counts["auxiliary"]],
        dtype=np.float64,
    )
    style_norm = np.linalg.norm(style_vec) + 1e-30
    style_vec = style_vec / style_norm
    style_hv = random_hv(d=dim, kind="real", seed=7) * style_vec.mean()
    bound = bind(bound, style_hv)

    proj = np.fft.fft(transformed, n=dim)
    final_repr = bound + proj

    return final_repr

def hybrid_decision(
    statistic: float,
    data: List[float],
    n: int,
    delta: float = 0.05,
    r: float = 1.0,
) -> Tuple[float, bool]:
    gini = gini_coefficient(data)
    alpha = 0.5 + 0.5 * gini 
    hoeffding_radius = hoeffding_bound(r, delta, n)
    scaled_radius = hoeffding_radius * alpha
    accept = statistic > scaled_radius
    return scaled_radius, accept

def improved_hybrid_representation(data: List[float], text: str, dim: int = 4096) -> np.ndarray:
    phash = compute_phash(data)
    bits = [(phash >> i) & 1 for i in reversed(range(64))]
    phash_vec = np.array([1.0 if b else -1.0 for b in bits], dtype=np.complex128)
    repeats = dim // len(phash_vec) + 1
    phash_vec = np.tile(phash_vec, repeats)[:dim]

    hv = random_hv(d=dim, kind="complex", seed=42)

    bound = bind(hv, phash_vec)

    gini = gini_coefficient(data)
    alpha = 0.5 + 0.5 * gini 

    transformed = np.array([fractional_power(x, alpha) for x in data], dtype=np.float64)

    style_counts = stylometry_analysis(text)
    style_vec = np.array(
        [style_counts["pronoun"], style_counts["article"],
         style_counts["preposition"], style_counts["auxiliary"]],
        dtype=np.float64,
    )
    style_norm = np.linalg.norm(style_vec) + 1e-30
    style_vec = style_vec / style_norm
    style_hv = random_hv(d=dim, kind="real", seed=7) * style_vec.mean()
    bound = bind(bound, style_hv)

    proj = np.fft.fft(transformed, n=dim)
    final_repr = bound + proj

    # Introduce a new normalization step to improve the stability of the representation
    final_repr = final_repr / np.linalg.norm(final_repr)

    return final_repr

def improved_hybrid_decision(
    statistic: float,
    data: List[float],
    n: int,
    delta: float = 0.05,
    r: float = 1.0,
) -> Tuple[float, bool]:
    gini = gini_coefficient(data)
    alpha = 0.5 + 0.5 * gini 
    hoeffding_radius = hoeffding_bound(r, delta, n)
    scaled_radius = hoeffding_radius * alpha
    accept = statistic > scaled_radius

    # Introduce a new thresholding mechanism to improve the accuracy of the decision
    threshold = 0.5
    confidence = 1 - (1 / (1 + math.exp(-statistic)))
    accept = accept and confidence > threshold

    return scaled_radius, accept