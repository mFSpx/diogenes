# DARWIN HAMMER — match 1093, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s3.py (gen2)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m62_s1.py (gen2)
# born: 2026-05-29T23:32:55Z

import hashlib
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Stylometry utilities (simplified)
# ----------------------------------------------------------------------
FUNCTION_CATS = {
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
    "conjunction": set("and but or nor so yet because although while".split()),
}


def stylometry_features(text: str) -> np.ndarray:
    """Return a fixed‑size numeric vector describing ``text``."""
    tokens = text.lower().split()
    n_words = len(tokens)
    n_chars = len(text)
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0

    # Category frequencies
    cat_counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1

    cat_freqs = [cat_counts[cat] / n_words if n_words else 0.0 for cat in sorted(FUNCTION_CATS)]

    return np.array([n_words, n_chars, avg_word_len, *cat_freqs], dtype=float)


# ----------------------------------------------------------------------
# Parent A – B‑spline basis (Cox‑de Boor recursion)
# ----------------------------------------------------------------------
def _cox_de_boor(x: float, k: int, i: int, knots: List[float]) -> float:
    """Recursive definition of B‑spline basis N_{i,k}(x)."""
    if k == 0:
        return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]
    term1 = 0.0
    term2 = 0.0
    if denom1 > 0:
        term1 = (x - knots[i]) / denom1 * _cox_de_boor(x, k - 1, i, knots)
    if denom2 > 0:
        term2 = (knots[i + k + 1] - x) / denom2 * _cox_de_boor(x, k - 1, i + 1, knots)
    return term1 + term2


def bspline_basis(x: float, knots: List[float], degree: int) -> np.ndarray:
    """Evaluate all B‑spline basis functions of given ``degree`` at ``x``."""
    n_basis = len(knots) - degree - 1
    return np.array([_cox_de_boor(x, degree, i, knots) for i in range(n_basis)], dtype=float)


# ----------------------------------------------------------------------
# Parent A – Single KAN layer with improved spline projection
# ----------------------------------------------------------------------
@dataclass
class KANLayer:
    """Parameters of a single KAN layer."""
    weight: np.ndarray          # shape (in_dim, out_dim)
    knots: List[np.ndarray]     # list of knot vectors, length = in_dim
    degree: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Map ``x`` (shape (in_dim,)) to ``y`` (shape (out_dim,))."""
        if x.ndim != 1 or x.shape[0] != self.weight.shape[0]:
            raise ValueError("Input dimension mismatch.")
        # Apply spline per input dimension
        spline_vals = []
        for xi, knot_vec in zip(x, self.knots):
            # Clamp xi to knot domain
            xi_clamped = min(max(xi, knot_vec[0]), knot_vec[-1] - 1e-12)
            spline_vals.append(bspline_basis(xi_clamped, knot_vec.tolist(), self.degree))
        # Stack to shape (in_dim, n_basis_i) – each row may have different length
        # For simplicity we project each spline vector onto a scalar via sum.
        # The KAN edge weight is then the scalar * weight.
        # This keeps everything in pure NumPy without variable‑length matrices.
        spline_scalars = np.array([np.linalg.norm(v) for v in spline_vals])  # (in_dim,)
        return spline_scalars @ self.weight  # (out_dim,)


def init_hybrid_layer(in_dim: int, out_dim: int, degree: int = 3, n_knots: int = 8) -> KANLayer:
    """Randomly initialise a KAN layer."""
    weight = np.random.randn(in_dim, out_dim) * 0.1
    knots = []
    for _ in range(in_dim):
        # Uniform knots on interval [0, 1] with extra padding for degree
        base = np.linspace(0.0, 1.0, n_knots)
        knot_vec = np.concatenate((
            np.full(degree, base[0]),
            base,
            np.full(degree, base[-1])
        ))
        knots.append(knot_vec)
    return KANLayer(weight=weight, knots=knots, degree=degree)


# ----------------------------------------------------------------------
# Parent B – Sparse WTA utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> np.ndarray:
    """Deterministically project ``values`` into an m‑dimensional sparse vector."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = np.zeros(m, dtype=float)
    for i, v in enumerate(values):
        for r in range(3):  # three hash repetitions per value
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            idx = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[idx] += sign * v
    return out


def top_k_mask(vec: np.ndarray, k: int) -> np.ndarray:
    """Return a binary mask of length ``len(vec)`` with 1 at the positions of the k
    largest (by absolute value) entries."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > vec.size:
        raise ValueError("k cannot exceed vector length")
    mask = np.zeros_like(vec, dtype=int)
    idx = np.argpartition(-np.abs(vec), k - 1)[:k]
    mask[idx] = 1
    return mask


def hamming(mask1: np.ndarray, mask2: np.ndarray) -> int:
    """Hamming distance between two binary masks."""
    if mask1.shape != mask2.shape:
        raise ValueError("Mask shapes differ")
    return int(np.sum(mask1 != mask2))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_feature_vector(text: str, kan: KANLayer) -> np.ndarray:
    """
    Convert raw ``text`` into a KAN‑processed feature vector.
    Steps:
        1. Stylometry extraction → s ∈ ℝᵈ
        2. KAN layer mapping    → y ∈ ℝᵐ

    Args:
        text (str): Input text
        kan (KANLayer): KAN layer instance

    Returns:
        np.ndarray: Hybrid feature vector
    """
    stylometry_vec = stylometry_features(text)
    kan_vec = kan(stylometry_vec)
    return kan_vec


def hybrid_privacy_model(text: str, kan: KANLayer, m: int, k: int) -> float:
    """
    Evaluate the hybrid privacy model for a given text and KAN layer.

    Args:
        text (str): Input text
        kan (KANLayer): KAN layer instance
        m (int): Dimensionality of sparse vector
        k (int): Number of top entries to select

    Returns:
        float: Privacy risk factor
    """
    kan_vec = hybrid_feature_vector(text, kan)
    sparse_vec = expand(kan_vec.tolist(), m)
    mask = top_k_mask(sparse_vec, k)
    ref_mask = top_k_mask(np.random.randn(m), k)
    hamming_dist = hamming(mask, ref_mask)
    risk_factor = hamming_dist / m
    return risk_factor


def main():
    # Example usage:
    text = "This is an example sentence."
    kan = init_hybrid_layer(7, 10)  # Initialize a KAN layer with 7 inputs and 10 outputs
    risk_factor = hybrid_privacy_model(text, kan, 100, 10)  # Evaluate the hybrid privacy model
    print(f"Risk factor: {risk_factor}")


if __name__ == "__main__":
    main()