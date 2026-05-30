# DARWIN HAMMER — match 1093, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s3.py (gen2)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m62_s1.py (gen2)
# born: 2026-05-29T23:32:55Z

"""Hybrid Algorithm Fusing Stylometry-KAN Model with Sparse WTA Encoding
====================================================================

This module integrates two distinct parent algorithms:

* **Parent A** – `hybrid_hybrid_hard_truth_ma_kan_m27_s3.py` provides a
  stylometric feature extraction and Kolmogorov-Arnold Networks (KAN) where every
  edge carries a learnable univariate B-spline.
* **Parent B** – `hybrid_sparse_wta_hybrid_privacy_model_m62_s1.py` implements
  Sparse Winner-Take-All (WTA) encoding with privacy-aware model-pool management.

**Mathematical bridge** – The stylometric vector `s ∈ ℝ^d` is a continuous
representation of a piece of text. The Sparse WTA encoding `expand(s, m)` maps
`s` into a high-dimensional sparse vector `e ∈ ℝ^m`. By feeding `e` into a
privacy-risk assessment function, we obtain a unified system that maps raw text
→ stylometric features → sparse encoding → privacy risk.

The code below implements:
* stylometric extraction and KAN layer (`stylometry_kan_layer`);
* Sparse WTA encoding (`sparse_wta_encode`);
* hybrid pipeline (`hybrid_pipeline`).

All operations are pure NumPy and rely only on the Python standard library.
"""

import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Stylometry and KAN utilities
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
        "and but or nor so yet because although".split()
    ),
}

def stylometry_features(text: str) -> np.ndarray:
    words = text.split()
    features = np.zeros(len(FUNCTION_CATS))
    for word in words:
        for cat, words_in_cat in FUNCTION_CATS.items():
            if word in words_in_cat:
                features[list(FUNCTION_CATS.keys()).index(cat)] += 1
    return features

def bspline_basis(x: float, k: int, t: np.ndarray) -> float:
    if k == 0:
        return 1.0 if t[0] <= x < t[1] else 0.0
    else:
        a = (x - t[0]) / (t[k] - t[0]) * bspline_basis(x, k-1, t)
        b = (t[k+1] - x) / (t[k+1] - t[1]) * bspline_basis(x, k-1, t)
        return a + b

def kan_layer(input_features: np.ndarray, weights: np.ndarray, t: np.ndarray) -> np.ndarray:
    output = np.zeros(len(weights))
    for i, (w, t_i) in enumerate(zip(weights, t)):
        output[i] = np.sum([w[j] * bspline_basis(input_features[j], 3, t_i) for j in range(len(input_features))])
    return output

# ----------------------------------------------------------------------
# Sparse WTA utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = '') -> List[float]:
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def top_k_mask(values: List[float], k: int) -> np.ndarray:
    indices = np.argsort(values)[::-1]
    mask = np.zeros(len(values))
    mask[indices[:k]] = 1.0
    return mask

def hamming_distance(mask1: np.ndarray, mask2: np.ndarray) -> int:
    return np.sum(np.abs(mask1 - mask2))

# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def stylometry_kan_layer(text: str, weights: np.ndarray, t: np.ndarray) -> np.ndarray:
    features = stylometry_features(text)
    return kan_layer(features, weights, t)

def sparse_wta_encode(values: List[float], m: int) -> np.ndarray:
    encoded = expand(values, m)
    return np.array(encoded)

def hybrid_pipeline(text: str, m: int, k: int, weights: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, int]:
    kan_output = stylometry_kan_layer(text, weights, t)
    encoded = sparse_wta_encode(kan_output.tolist(), m)
    mask = top_k_mask(encoded, k)
    distance = hamming_distance(mask, np.zeros_like(mask))
    return mask, distance

if __name__ == "__main__":
    text = "This is an example sentence."
    m = 100
    k = 10
    weights = np.random.rand(5)
    t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    mask, distance = hybrid_pipeline(text, m, k, weights, t)
    print(mask)
    print(distance)