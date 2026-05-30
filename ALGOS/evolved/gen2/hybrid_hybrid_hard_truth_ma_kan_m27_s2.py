# DARWIN HAMMER — match 27, survivor 2
# gen: 2
# parent_a: hybrid_hard_truth_math_model_pool_m8_s4.py (gen1)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:25:23Z

"""
This module fuses the stylometry analysis from 'hybrid_hard_truth_math_model_pool_m8_s4.py' 
with the Kolmogorov-Arnold Networks (KAN) implementation from 'kan.py'. 
The mathematical bridge between the two structures lies in the use of B-splines 
in KANs to approximate univariate functions, which can be applied to the 
stylometry features extracted from text data. 
By integrating the stylometry analysis into the KAN architecture, we can 
leverage the power of deep learning to improve the accuracy of stylometry-based 
text classification tasks.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import math
import random
import sys
import pathlib

# Stylometry utilities
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
        sum(1 for ch in text if ch.isupper()) / total_chars
    ]
    return np.array(vals)

# KAN utilities
def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector: repeat boundary knots (k-1) times so that
    # the polynomial spans cleanly to the boundary.
    knot_vector = np.concatenate((np.full(k-1, grid[0]), grid, np.full(k-1, grid[-1])))

    # Initialize B-spline basis functions
    B = np.zeros((len(x), len(grid) - 1))

    # Compute B-spline basis functions
    for i, xi in enumerate(x):
        for j, gj in enumerate(grid[:-1]):
            B[i, j] = bspline(xi, knot_vector, k, j)

    return B

def bspline(x: float, knot_vector: np.ndarray, k: int, j: int) -> float:
    if k == 1:
        if knot_vector[j] <= x < knot_vector[j+1]:
            return 1.0
        else:
            return 0.0
    else:
        d1 = knot_vector[j+k] - knot_vector[j]
        d2 = knot_vector[j+k+1] - knot_vector[j+1]

        if d1 != 0:
            b1 = (x - knot_vector[j]) / d1
        else:
            b1 = 0.0

        if d2 != 0:
            b2 = (knot_vector[j+k+1] - x) / d2
        else:
            b2 = 0.0

        return b1 * bspline(x, knot_vector, k-1, j) + b2 * bspline(x, knot_vector, k-1, j+1)

def kan_layer(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    B = bspline_basis(x, grid, k)
    return B.dot(np.random.rand(B.shape[1]))

# Hybrid functions
def stylometry_kan(text: str, grid: np.ndarray, k: int = 3) -> np.ndarray:
    features = stylometry_features(text)
    return kan_layer(features, grid, k)

def kan_stylometry(x: np.ndarray, grid: np.ndarray, k: int = 3) -> dict[str, float]:
    features = kan_layer(x, grid, k)
    stylometry_features_list = [features]
    total = max(1, len(stylometry_features_list))
    cnt = Counter(stylometry_features_list)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def hybrid_text_analysis(text: str, grid: np.ndarray, k: int = 3) -> Tuple[np.ndarray, dict[str, float]]:
    kan_output = stylometry_kan(text, grid, k)
    stylometry_output = kan_stylometry(kan_output, grid, k)
    return kan_output, stylometry_output

if __name__ == "__main__":
    text = "This is a sample text."
    grid = np.linspace(0, 1, 10)
    kan_output, stylometry_output = hybrid_text_analysis(text, grid)
    print(kan_output)
    print(stylometry_output)