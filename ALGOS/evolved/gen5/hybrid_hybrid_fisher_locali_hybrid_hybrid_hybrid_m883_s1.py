# DARWIN HAMMER — match 883, survivor 1
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s3.py (gen4)
# born: 2026-05-29T23:31:33Z

"""
Hybrid Fisher‑Stylometry Fusion

Parents:
- hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s2.py (Algorithm A)
- hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s3.py (Algorithm B)

Mathematical bridge:
Algorithm A supplies a scalar Fisher‑information score derived from a Gaussian beam:
    I(θ) = (∂_θ g(θ))² / g(θ) ,   g(θ)=exp(‑½((θ‑c)/w)²)
Algorithm B supplies a high‑dimensional stylometric vector **s** ∈ ℝᴰ (D≈96).

The fusion treats each component s_i as an “angle” θ_i and evaluates the
Fisher‑information I(s_i; c, w) with a centre c = mean(s) and width w = std(s).
The resulting scalar I_i is stacked with s_i to form a 2‑column path
P = [(s_1, I_1), …, (s_D, I_D)].
A lead‑lag transform (Algorithm A) is then applied to P, producing a
(2·D‑1)×4 matrix that interleaves “lead” (s, I) and “lag” (s_next, I_current)
channels.  This matrix can be used for downstream similarity or classification
tasks, thus mathematically fusing the Fisher‑information scoring with the
stylometric feature extraction.

The module provides three public functions demonstrating the hybrid operation:
1. `stylometry_features(text, dim=96)` – extracts a D‑dimensional vector.
2. `fisher_features_from_vector(vec, center=None, width=None)` – computes the
   Fisher‑information for each component.
3. `hybrid_fisher_stylometry_path(text, dim=96)` – builds the lead‑lag matrix
   from a raw text string.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Gaussian beam & Fisher information
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam g(θ) = exp(-½ ((θ‑c)/w)²)."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher‑information I(θ) = (∂_θ g)² / g, with safe guard eps."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a (T, d) path → (2·T‑1, 2·d) matrix.
    For each time step t we emit [x_t, x_t] then [x_{t+1}, x_t].
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


# ----------------------------------------------------------------------
# Parent B – Stylometric extraction
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
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def words(text: str) -> List[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Low‑dimensional semantic (LDS) vector: proportion of function‑category words.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a fixed‑size numeric vector from raw text.
    Combines handcrafted ratios with the LDS vector and pads/truncates to `dim`.
    """
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    handcrafted = np.array(
        [
            total_words / 500.0,
            sum(len(w) for w in ws) / total_words / 12.0,
            (text.count("\n") + 1) / 200.0,
            sum(text.count(p) for p in "!?") / total_chars,
            sum(text.count(p) for p in ";:") / total_chars,
            sum(text.count(p) for p in "-—") / total_chars,
        ],
        dtype=float,
    )

    lsm = np.array(list(lsm_vector(text).values()), dtype=float)

    # Concatenate and then project to the requested dimensionality via a simple linear map.
    raw = np.concatenate([handcrafted, lsm])
    if raw.size == dim:
        return raw
    elif raw.size < dim:
        # Pad with zeros
        padded = np.zeros(dim, dtype=float)
        padded[: raw.size] = raw
        return padded
    else:
        # Random but reproducible projection (seed based on dim)
        rng = np.random.default_rng(seed=dim)
        proj = rng.standard_normal((raw.size, dim))
        return raw @ proj  # shape (dim,)


# ----------------------------------------------------------------------
# Fusion layer – Fisher‑information on stylometric dimensions
# ----------------------------------------------------------------------
def fisher_features_from_vector(
    vec: np.ndarray, center: float | None = None, width: float | None = None
) -> np.ndarray:
    """
    Compute Fisher‑information for each component of `vec`.
    If `center`/`width` are omitted they are taken as the mean and standard deviation
    of the vector (with a small epsilon for stability).
    Returns an array of the same shape as `vec`.
    """
    if vec.ndim != 1:
        raise ValueError("vec must be 1‑D")
    if center is None:
        center = float(np.mean(vec))
    if width is None:
        width = float(np.std(vec)) + 1e-9  # avoid zero
    # Vectorised version of fisher_score
    z = (vec - center) / width
    intensity = np.exp(-0.5 * z ** 2)
    intensity = np.maximum(intensity, 1e-12)
    derivative = intensity * (-(vec - center) / (width ** 2))
    return (derivative ** 2) / intensity


def hybrid_fisher_stylometry_path(text: str, dim: int = 96) -> np.ndarray:
    """
    Full hybrid pipeline:
    1. Extract a D‑dimensional stylometric vector `s`.
    2. Compute per‑component Fisher‑information `i`.
    3. Stack into a (D, 2) path and apply the lead‑lag transform.
    Returns the (2·D‑1, 4) matrix.
    """
    s = stylometry_features(text, dim=dim)
    i = fisher_features_from_vector(s)
    path = np.column_stack((s, i))          # shape (D, 2)
    return lead_lag_transform(path)        # shape (2·D‑1, 4)


def hybrid_distance(text_a: str, text_b: str, dim: int = 96) -> float:
    """
    Example similarity metric:
    Euclidean distance between the lead‑lag matrices of two texts,
    after flattening to a 1‑D vector (zero‑padding the shorter one).
    """
    mat_a = hybrid_fisher_stylometry_path(text_a, dim=dim)
    mat_b = hybrid_fisher_stylometry_path(text_b, dim=dim)

    # Flatten
    flat_a = mat_a.ravel()
    flat_b = mat_b.ravel()

    # Pad to equal length
    max_len = max(flat_a.size, flat_b.size)
    if flat_a.size < max_len:
        flat_a = np.pad(flat_a, (0, max_len - flat_a.size))
    if flat_b.size < max_len:
        flat_b = np.pad(flat_b, (0, max_len - flat_b.size))

    return float(np.linalg.norm(flat_a - flat_b))


# ----------------------------------------------------------------------
# Optional data structure from Parent B (demonstrates integration)
# ----------------------------------------------------------------------
@dataclass
class HybridNode:
    """
    Tree node that stores the raw text, its stylometric vector,
    and the lead‑lag matrix for quick downstream graph‑based analysis.
    """
    text: str
    vector: np.ndarray
    lead_lag_matrix: np.ndarray

    @classmethod
    def from_text(cls, text: str, dim: int = 96) -> "HybridNode":
        vec = stylometry_features(text, dim=dim)
        ll = hybrid_fisher_stylometry_path(text, dim=dim)
        return cls(text=text, vector=vec, lead_lag_matrix=ll)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_a = "The quick brown fox jumps over the lazy dog. It was a bright sunny day."
    sample_b = "In a distant galaxy, the starship Voyager glided silently through the nebula."

    print("Stylometry vector (dim=96) for sample A:", stylometry_features(sample_a).shape)
    print("Fisher features (same dim):", fisher_features_from_vector(stylometry_features(sample_a)).shape)

    ll_matrix = hybrid_fisher_stylometry_path(sample_a)
    print("Lead‑lag matrix shape:", ll_matrix.shape)

    dist = hybrid_distance(sample_a, sample_b)
    print("Hybrid Euclidean distance between samples:", dist)

    node = HybridNode.from_text(sample_a)
    print("HybridNode created – lead‑lag matrix first row:", node.lead_lag_matrix[0])