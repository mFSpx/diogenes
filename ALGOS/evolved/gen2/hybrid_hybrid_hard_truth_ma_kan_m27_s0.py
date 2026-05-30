# DARWIN HAMMER — match 27, survivor 0
# gen: 2
# parent_a: hybrid_hard_truth_math_model_pool_m8_s4.py (gen1)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:25:23Z

"""
Fusion of the DARWIN HAMMER (hybrid_hard_truth_math_model_pool_m8_s4.py) and the 
Kolmogorov-Arnold Networks (KAN) algorithm (kan.py). The mathematical bridge 
between the two structures is found by integrating the stylometry and LSM 
vector operations of the DARWIN HAMMER with the B-spline basis and deep KAN 
composition of the KAN algorithm. Specifically, the stylometry and LSM 
vector operations are used to generate input features for the KAN layers, 
which are then used to predict the output.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]


def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    import hashlib
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))
    vals: List[float] = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_chars,
    ]
    return np.array(vals)


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------
def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector: repeat boundary knots (k-1) times so that
    # the polynomial spans cleanly to the boundary.
    knots = np.zeros(len(grid) + 2 * (k - 1), dtype=np.float64)
    knots[:k - 1] = grid[0]
    knots[k - 1:-k + 1] = grid
    knots[-k + 1:] = grid[-1]

    # Initialize basis functions
    B = np.zeros((len(x), len(grid) - 1), dtype=np.float64)

    # Evaluate B-spline basis functions
    for i in range(len(x)):
        for j in range(len(grid) - 1):
            for r in range(k):
                if knots[j + r] <= x[i] < knots[j + r + 1]:
                    if r == 0:
                        B[i, j] = (x[i] - knots[j]) / (knots[j + k] - knots[j])
                    elif r == k - 1:
                        B[i, j] = (knots[j + k + 1] - x[i]) / (knots[j + k + 1] - knots[j + 1])
                    else:
                        B[i, j] = (x[i] - knots[j + r]) / (knots[j + r + 1] - knots[j + r]) * B[i, j - 1] + (knots[j + r + k + 1] - x[i]) / (knots[j + r + k + 1] - knots[j + r + 1]) * B[i, j]

    return B


# ---------------------------------------------------------------------------
# KAN layer
# ---------------------------------------------------------------------------
def kan_layer(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    return np.dot(bspline_basis(x, grid, k), np.random.rand(len(grid) - 1))


# ---------------------------------------------------------------------------
# Hybrid operation
# ---------------------------------------------------------------------------
def hybrid_operation(text: str, grid: np.ndarray, k: int = 3) -> np.ndarray:
    stylometry = stylometry_features(text)
    return kan_layer(stylometry, grid, k)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------
def test_stylometry_features():
    text = "This is a test text."
    stylometry = stylometry_features(text)
    assert len(stylometry) == 6


def test_bspline_basis():
    x = np.array([0.5, 0.7, 0.3])
    grid = np.array([0.2, 0.4, 0.6, 0.8])
    basis = bspline_basis(x, grid)
    assert basis.shape == (3, 3)


def test_hybrid_operation():
    text = "This is a test text."
    grid = np.array([0.2, 0.4, 0.6, 0.8])
    result = hybrid_operation(text, grid)
    assert result.shape == (6,)


if __name__ == "__main__":
    test_stylometry_features()
    test_bspline_basis()
    test_hybrid_operation()