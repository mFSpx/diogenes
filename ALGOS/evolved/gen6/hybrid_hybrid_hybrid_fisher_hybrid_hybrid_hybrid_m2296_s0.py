# DARWIN HAMMER — match 2296, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_korpus_text_h_m537_s0.py (gen5)
# born: 2026-05-29T23:41:47Z

"""
Hybrid Fusion Module: hybrid_fisher_locali_hybrid_geomet_endpoint_regret.py

Parents:
- hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_endpoi_hybrid_korpus_text_h_m537_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A supplies a Fisher‑information scalar F(θ) that quantifies the
sensitivity of a Gaussian beam intensity I(θ).  This scalar can be promoted
to a diagonal weighting matrix **W** that modulates any vector‑ or multivector‑
valued quantity.

Algorithm B yields a morphology‑derived recovery priority p∈[0,1] and a
text‑derived feature score vector **s**∈ℝ⁹.  The priority acts as a global
scalar that attenuates the decision scores.

The hybrid system therefore builds a *double‑weighted* construct:

    **H** = p · (**W** ⊙ **G**)·**s**

where **G** is the geometric‑product matrix obtained from two multivectors,
⊙ denotes element‑wise (Hadamard) multiplication with the Fisher weighting
matrix **W**, and the resulting vector is finally scaled by the morphology
priority p.  This unifies the Fisher‑metric, Clifford‑geometric product, and
morphology‑aware decision scoring into a single mathematically coherent pipeline.
"""

import math
import re
import sys
import random
import pathlib
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Fisher information & geometric product utilities
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I   where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort indices and cancel pairs of equal indices (which square to +1).
    Returns the sorted tuple of remaining indices and the sign (+1 / -1)
    induced by the swaps needed to bring the original list into the sorted order.
    """
    # Count occurrences
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # Keep only indices that appear an odd number of times
    remaining = [i for i, c in counts.items() if c % 2 == 1]

    # Determine sign by counting swaps required to sort the original list
    # Simple bubble‑sort sign computation
    sign = 1
    lst = list(indices)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                sign = -sign
    return tuple(sorted(remaining)), sign


def geometric_product(a: Dict[Tuple[int, ...], float],
                     b: Dict[Tuple[int, ...], float]) -> Dict[Tuple[int, ...], float]:
    """
    Compute the Clifford geometric product of two multivectors `a` and `b`.
    Multivectors are represented as dicts mapping blade index tuples to scalars.
    """
    result: Dict[Tuple[int, ...], float] = {}
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            merged = blade_a + blade_b
            cleaned, sign = _blade_sign(merged)
            coeff = coeff_a * coeff_b * sign
            result[cleaned] = result.get(cleaned, 0.0) + coeff
    return result


def multivector_to_array(mv: Dict[Tuple[int, ...], float],
                         max_grade: int = 3) -> np.ndarray:
    """
    Flatten a multivector into a dense NumPy array of length 2**max_grade.
    The ordering follows binary encoding of basis blades.
    """
    size = 2 ** max_grade
    arr = np.zeros(size, dtype=float)
    for blade, coeff in mv.items():
        # Encode blade as integer mask
        mask = 0
        for idx in blade:
            mask |= 1 << idx
        if mask < size:
            arr[mask] = coeff
    return arr


def fisher_weight_matrix(thetas: List[float],
                         centers: List[float],
                         widths: List[float]) -> np.ndarray:
    """
    Build a diagonal matrix W where W_ii = fisher_score(thetas[i], centers[i], widths[i]).
    """
    if not (len(thetas) == len(centers) == len(widths)):
        raise ValueError("Input lists must have the same length")
    scores = [fisher_score(t, c, w) for t, c, w in zip(thetas, centers, widths)]
    return np.diag(scores)


# ----------------------------------------------------------------------
# Algorithm B – Morphology priority & text feature scoring
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Maps righting time index to a priority in [0,1]:
        p = min(1, R / max_index)
    """
    r = righting_time_index(m)
    return min(1.0, r / max_index)


def extract_text_features(text: str) -> np.ndarray:
    """
    Produce a 9‑dimensional feature vector from regex counts.
    The patterns are deliberately simple placeholders.
    """
    patterns = [
        r"\bthe\b", r"\band\b", r"\bto\b", r"\bof\b", r"\b[a-z]{4,}\b",
        r"\d+", r"[A-Z][a-z]+", r"\s+", r"[!?.]"
    ]
    counts = [len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns]
    vec = np.array(counts, dtype=float)
    # Normalize to unit L2 norm to keep scale reasonable
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def weighted_text_score(features: np.ndarray,
                       pos_weights: np.ndarray,
                       neg_weights: np.ndarray) -> np.ndarray:
    """
    Apply positive and negative weight matrices to the feature vector.
    The result is a 9‑dimensional score vector.
    """
    if pos_weights.shape != (9, 9) or neg_weights.shape != (9, 9):
        raise ValueError("Weight matrices must be 9x9")
    pos_part = pos_weights @ features
    neg_part = neg_weights @ features
    return pos_part - neg_part


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------


def hybrid_fisher_geometric(thetas: List[float],
                            centers: List[float],
                            widths: List[float],
                            mv_a: Dict[Tuple[int, ...], float],
                            mv_b: Dict[Tuple[int, ...], float]) -> np.ndarray:
    """
    Compute the Fisher‑weighted geometric product of two multivectors.
    Returns a dense NumPy array representing the weighted multivector.
    """
    # 1. Geometric product
    gp = geometric_product(mv_a, mv_b)

    # 2. Flatten to array (use grade up to 3 → length 8)
    gp_arr = multivector_to_array(gp, max_grade=3)

    # 3. Build Fisher diagonal matrix (size must match gp_arr length)
    W = fisher_weight_matrix(thetas, centers, widths)
    if W.shape[0] != gp_arr.shape[0]:
        # Pad or truncate to match dimensions
        size = max(W.shape[0], gp_arr.shape[0])
        W_padded = np.zeros((size, size))
        W_padded[:W.shape[0], :W.shape[1]] = W
        W = W_padded
        gp_arr = np.pad(gp_arr, (0, size - gp_arr.shape[0]), 'constant')
    # Element‑wise weighting via matrix multiplication (since W is diagonal)
    weighted = W @ gp_arr
    return weighted


def hybrid_morphology_text(m: Morphology,
                           text: str,
                           pos_weights: np.ndarray,
                           neg_weights: np.ndarray) -> np.ndarray:
    """
    Produce a morphology‑aware decision vector from raw text.
    Steps:
        1. Extract normalized text features.
        2. Apply positive/negative weight matrices.
        3. Scale by recovery priority p.
    """
    p = recovery_priority(m)
    feats = extract_text_features(text)
    raw_score = weighted_text_score(feats, pos_weights, neg_weights)
    return p * raw_score


def hybrid_combined(thetas: List[float],
                    centers: List[float],
                    widths: List[float],
                    mv_a: Dict[Tuple[int, ...], float],
                    mv_b: Dict[Tuple[int, ...], float],
                    m: Morphology,
                    text: str,
                    pos_weights: np.ndarray,
                    neg_weights: np.ndarray) -> np.ndarray:
    """
    Full hybrid pipeline:
        H = p · (W ⊙ G) · s
    where
        G = geometric_product(mv_a, mv_b) flattened,
        W = Fisher diagonal matrix,
        p = recovery priority,
        s = weighted text score vector.
    The final result is a 9‑dimensional vector (truncated/padded as needed).
    """
    # Fisher‑weighted geometric product (vector length may differ from 9)
    weighted_geo = hybrid_fisher_geometric(thetas, centers, widths, mv_a, mv_b)

    # Morphology‑aware text score (length 9)
    txt_score = hybrid_morphology_text(m, text, pos_weights, neg_weights)

    # Align dimensions: pad the shorter vector with zeros
    max_len = max(weighted_geo.shape[0], txt_score.shape[0])
    geo_padded = np.pad(weighted_geo, (0, max_len - weighted_geo.shape[0]), 'constant')
    txt_padded = np.pad(txt_score, (0, max_len - txt_score.shape[0]), 'constant')

    # Element‑wise multiplication then sum to produce final vector
    combined = geo_padded * txt_padded
    # Optionally return the combined vector itself (still length max_len)
    return combined


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data for the test
    thetas = [0.1, 0.5, 1.0]
    centers = [0.0, 0.0, 0.0]
    widths = [0.2, 0.2, 0.2]

    # Simple multivectors in 3‑dimensional Clifford algebra (basis e0, e1, e2)
    mv1 = {(): 1.0, (0,): 0.5, (1, 2): -0.3}
    mv2 = {(): 2.0, (1,): 0.7, (0, 2): 0.4}

    # Morphology instance
    morph = Morphology(length=5.0, width=2.0, height=1.5, mass=3.0)

    # Dummy text
    sample_text = "The quick brown fox jumps over the lazy dog. And then it runs to the hill!"

    # Random positive/negative weight matrices (ensure they are 9x9)
    rng = np.random.default_rng(42)
    pos_w = rng.random((9, 9))
    neg_w = rng.random((9, 9))

    # Run the full hybrid pipeline
    result = hybrid_combined(thetas, centers, widths,
                             mv1, mv2,
                             morph, sample_text,
                             pos_w, neg_w)

    print("Hybrid combined vector (length {}):".format(result.shape[0]))
    print(result)