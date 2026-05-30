# DARWIN HAMMER — match 27, survivor 1
# gen: 2
# parent_a: hybrid_hard_truth_math_model_pool_m8_s4.py (gen1)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:25:23Z

"""
This module implements a hybrid algorithm that combines the stylometry and LSM utilities from the 'hybrid_hard_truth_math_model_pool_m8_s4' algorithm with the Kolmogorov-Arnold Networks (KAN) from the 'kan' algorithm.

The mathematical bridge between the two algorithms is found in the representation of the stylometry features as a continuous multivariate function, which can be approximated using the KAN architecture. Specifically, the stylometry features are used as the input to the KAN, and the output of the KAN is used to compute the stylometry features.

This hybrid algorithm allows for the representation of stylometry features in a more flexible and powerful way, using the KAN architecture to learn complex relationships between the input features.
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
    return np.array(vals + [0.0] * (dim - len(vals)))


# ----------------------------------------------------------------------
# Parent B – Kolmogorov-Arnold Networks (KAN)
# ----------------------------------------------------------------------
def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector: repeat boundary knots (k-1) times so that
    # the polynomial spans cleanly to the boundary.
    knots = np.array([grid[0]] * (k - 1) + list(grid) + [grid[-1]] * (k - 1))
    n_basis = len(grid) - 1
    n_knots = len(knots)

    # Initialize the basis matrix
    B = np.zeros((len(x), n_basis))

    # Compute the basis functions
    for i in range(n_basis):
        for j in range(len(x)):
            B[j, i] = bspline(x[j], knots, i, k)

    return B


def bspline(x: float, knots: np.ndarray, i: int, k: int) -> float:
    if k == 1:
        if knots[i] <= x < knots[i + 1]:
            return 1.0
        else:
            return 0.0
    else:
        d1 = knots[i + k - 1] - knots[i]
        d2 = knots[i + k] - knots[i + 1]
        e1 = 0.0 if d1 == 0.0 else (x - knots[i]) / d1
        e2 = 0.0 if d2 == 0.0 else (knots[i + k] - x) / d2
        return e1 * bspline(x, knots, i, k - 1) + e2 * bspline(x, knots, i + 1, k - 1)


def kan_layer(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    B = bspline_basis(x, grid, k)
    return np.sum(B, axis=1)


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    stylometry_vals = stylometry_features(text, dim)
    kan_vals = kan_layer(stylometry_vals, np.linspace(0, 1, dim))
    return np.concatenate((stylometry_vals, kan_vals))


def hybrid_lsm_vector(text: str) -> dict[str, float]:
    lsm_vals = lsm_vector(text)
    kan_vals = kan_layer(np.array(list(lsm_vals.values())), np.linspace(0, 1, len(lsm_vals)))
    return {cat: val + kan_vals[0] for cat, val in lsm_vals.items()}


def hybrid_stable_hash(text: str) -> int:
    stylometry_vals = stylometry_features(text)
    kan_vals = kan_layer(stylometry_vals, np.linspace(0, 1, len(stylometry_vals)))
    return stable_hash(str(np.concatenate((stylometry_vals, kan_vals))))


if __name__ == "__main__":
    text = "This is a test text."
    print(hybrid_stylometry_features(text))
    print(hybrid_lsm_vector(text))
    print(hybrid_stable_hash(text))