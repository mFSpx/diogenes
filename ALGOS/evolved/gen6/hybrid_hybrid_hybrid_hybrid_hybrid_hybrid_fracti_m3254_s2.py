# DARWIN HAMMER — match 3254, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m552_s1.py (gen5)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s1.py (gen2)
# born: 2026-05-29T23:50:15Z

"""Hybrid Fusion of Perceptual Hash / Stylometry with Fractional Hoeffding Mechanics.

Parents:
- hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m552_s1.py (Algorithm A)
- hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s1.py (Algorithm B)

Mathematical Bridge:
The bridge is built on two shared quantitative notions:
1. **Distribution inequality** – measured by the Gini coefficient on a numeric
   feature vector.  The Gini value is used as a *scale* for the fractional
   exponent α in a power‑law transformation (Algorithm B) and simultaneously
   as a weighting factor for the perceptual hash‑derived hypervector (Algorithm A).
2. **Confidence scaling** – the Hoeffding bound provides a data‑driven confidence
   radius.  By multiplying this radius with the Gini‑scaled α we obtain a
   unified uncertainty term that modulates both the RBF surrogate prediction
   (from Algorithm A) and the bound‑based decision rule (from Algorithm B).

The resulting hybrid system therefore:
- hashes numeric data (perceptual hash),
- binds the hash to a random high‑dimensional hypervector,
- reshapes the data with a Gini‑scaled fractional power,
- feeds the transformed data to a lightweight RBF surrogate,
- and finally makes a Hoeffding‑bound‑aware decision.

All core equations from the parents are retained and mathematically intertwined.
"""

import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]


def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a numeric sequence."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def stylometry_analysis(text: str) -> Dict[str, int]:
    """Simple stylometry: count occurrences of functional word categories."""
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
    """Generate a random hypervector."""
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
    """Circular convolution binding via FFT (complex hypervectors)."""
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Inverse binding (deconvolution) using complex conjugate."""
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for bounded random variables in [0, r]."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Compute Gini coefficient (inequality) of a non‑negative distribution."""
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
    """Signed fractional power: sign(x) * |x|**alpha."""
    if x == 0.0:
        return 0.0
    return math.copysign(abs(x) ** alpha, x)


def rbf_kernel(a: np.ndarray, b: np.ndarray, epsilon: float = 1.0) -> float:
    """Radial‑Basis‑Function kernel (Gaussian) between two vectors."""
    diff = a - b
    return math.exp(-epsilon * np.dot(diff, diff))


def rbf_surrogate_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    query: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """Lightweight RBF surrogate: weighted average of training targets."""
    if train_X.shape[0] != train_y.shape[0]:
        raise ValueError("train_X and train_y must have same length")
    kernels = np.array([rbf_kernel(query, xi, epsilon) for xi in train_X])
    if kernels.sum() == 0:
        return float(np.mean(train_y))
    weights = kernels / kernels.sum()
    return float(np.dot(weights, train_y))


def hybrid_representation(data: List[float], text: str, dim: int = 4096) -> np.ndarray:
    """
    Fuse perceptual hashing, stylometry, Gini‑scaled fractional power,
    and hypervector binding into a single high‑dimensional representation.
    """
    # 1. Perceptual hash -> binary +/-1 vector
    phash = compute_phash(data)
    bits = [(phash >> i) & 1 for i in reversed(range(64))]
    phash_vec = np.array([1.0 if b else -1.0 for b in bits], dtype=np.complex128)
    # Pad / repeat to match desired dimension
    repeats = dim // len(phash_vec) + 1
    phash_vec = np.tile(phash_vec, repeats)[:dim]

    # 2. Random hypervector
    hv = random_hv(d=dim, kind="complex", seed=42)

    # 3. Bind hash with random hypervector
    bound = bind(hv, phash_vec)

    # 4. Gini coefficient as scaling factor for fractional exponent
    gini = gini_coefficient(data)
    alpha = 0.5 + 0.5 * gini  # α ∈ [0.5, 1.0]

    # 5. Apply fractional power element‑wise
    transformed = np.array([fractional_power(x, alpha) for x in data], dtype=np.float64)

    # 6. Incorporate stylometry counts (converted to a small vector)
    style_counts = stylometry_analysis(text)
    style_vec = np.array(
        [style_counts["pronoun"], style_counts["article"],
         style_counts["preposition"], style_counts["auxiliary"]],
        dtype=np.float64,
    )
    # Normalize and embed into the high‑dimensional space
    style_norm = np.linalg.norm(style_vec) + 1e-30
    style_vec = style_vec / style_norm
    style_hv = random_hv(d=dim, kind="real", seed=7) * style_vec.mean()
    bound = bind(bound, style_hv)

    # 7. Combine numeric transformed data (projected) with bound representation
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
    """
    Make a decision using a Hoeffding bound scaled by the Gini‑derived α.
    Returns (bound, accept) where accept is True if statistic exceeds the bound.
    """
    gini = gini_coefficient(data)
    alpha = 0.5 + 0.5 * gini
    bound = hoeffding_bound(r, delta, n) * alpha
    accept = statistic > bound
    return bound, accept


def hybrid_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    query_X: np.ndarray,
    data_for_gini: List[float],
    text: str,
) -> np.ndarray:
    """
    End‑to‑end hybrid prediction:
    1. Build hybrid representation for each query point.
    2. Feed representation to an RBF surrogate trained on the same representation.
    Returns predicted values for each query.
    """
    # Build representation for training set (once)
    train_repr = np.vstack(
        [hybrid_representation(row.tolist(), text) for row in train_X]
    )
    # Train surrogate (store representation & targets)
    # For simplicity we reuse the same RBF function directly.
    preds = np.empty(query_X.shape[0], dtype=float)
    for i, q in enumerate(query_X):
        q_repr = hybrid_representation(q.tolist(), text)
        preds[i] = rbf_surrogate_predict(train_repr, train_y, q_repr)
    return preds


if __name__ == "__main__":
    # Smoke test
    sample_data = [random.uniform(-5, 5) for _ in range(100)]
    sample_text = "I think that the quick brown fox jumps over the lazy dog while we observe."
    repr_vec = hybrid_representation(sample_data, sample_text, dim=2048)
    print("Hybrid representation shape:", repr_vec.shape)

    stat = sum(sample_data) / len(sample_data)
    bound, accept = hybrid_decision(stat, sample_data, n=len(sample_data))
    print(f"Statistic={stat:.4f}, Hoeffding‑scaled bound={bound:.4f}, accept={accept}")

    # Dummy training data for prediction
    X_train = np.random.randn(20, 10)
    y_train = np.random.randn(20)
    X_query = np.random.randn(5, 10)
    preds = hybrid_predict(X_train, y_train, X_query, sample_data, sample_text)
    print("Predictions:", preds)