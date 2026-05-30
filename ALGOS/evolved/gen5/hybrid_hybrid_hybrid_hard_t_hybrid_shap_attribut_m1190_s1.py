# DARWIN HAMMER — match 1190, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s3.py (gen3)
# parent_b: hybrid_shap_attribution_hybrid_hybrid_pherom_m70_s1.py (gen4)
# born: 2026-05-29T23:33:16Z

"""
Hybrid Stylometric-Geometric-Shapley Model
==========================================

This module fuses three previously independent algorithms:

* **Parent A** – a stylometric extractor that builds a high-dimensional frequency fingerprint (`stylometry_features`) from textual data (hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s3.py).
* **Parent B** – a geometric toolkit that constructs Voronoi partitions of n-dimensional points and provides a minimal Clifford-algebra blade implementation (hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s3.py).
* **Parent C** – a Shapley value calculator for feature attribution (hybrid_shap_attribution_hybrid_hybrid_pherom_m70_s1.py).

The mathematical bridge between Parent A and Parent B is the observation that a stylometric fingerprint is simply a point in ℝⁿ (n = 96 in the original implementation). 
The bridge between Parent A/B and Parent C is established by treating each feature in the Shapley calculation as a dimension in the stylometric fingerprint space.

By fusing these algorithms, we create a unified workflow: text → vector → geometric region & algebraic signature → feature attribution.
"""

import re
import hashlib
import math
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, FrozenSet, Iterable
import numpy as np

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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot".split()
    ),
}

def stylometry_features(text: str) -> np.ndarray:
    tokens = re.findall(r'\b\w+\b', text.lower())
    features = np.zeros(96)
    for token in tokens:
        if token in FUNCTION_CATS["pronoun"]:
            features[0] += 1
        elif token in FUNCTION_CATS["article"]:
            features[1] += 1
        elif token in FUNCTION_CATS["preposition"]:
            features[2:10] += 1
        elif token in FUNCTION_CATS["auxiliary"]:
            features[10:20] += 1
        elif token in FUNCTION_CATS["conjunction"]:
            features[20:30] += 1
        elif token in FUNCTION_CATS["negation"]:
            features[30] += 1
    return features / len(tokens)

def voronoi_partition(points: np.ndarray, seed: np.ndarray) -> int:
    distances = np.linalg.norm(points - seed, axis=1)
    return np.argmin(distances)

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)

def compute_dhash(values: list[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def shap_value(feature_index: int, feature_count: int, value_fn: callable) -> float:
    total = 0.0
    for k in range(feature_count + 1):
        for subset in get_subsets(range(feature_count), k):
            s = frozenset(subset)
            total += shapley_kernel_weight(k, feature_count) * (value_fn(s | {feature_index}) - value_fn(s))
    return total

def get_subsets(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - r + i:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def hybrid_stylometric_shapley(text: str, feature_count: int) -> Dict[int, float]:
    features = stylometry_features(text)
    shap_values = {}
    for i in range(feature_count):
        def value_fn(subset: FrozenSet[int]) -> float:
            subset_features = features.copy()
            for j in subset:
                subset_features[j] = 0
            return np.linalg.norm(subset_features)
        shap_values[i] = shap_value(i, feature_count, value_fn)
    return shap_values

def hybrid_voronoi_shapley(points: np.ndarray, seed: np.ndarray, feature_count: int) -> Dict[int, float]:
    voronoi_cell = voronoi_partition(points, seed)
    shap_values = {}
    for i in range(feature_count):
        def value_fn(subset: FrozenSet[int]) -> float:
            subset_points = points.copy()
            for j in subset:
                subset_points[j] = seed
            return np.linalg.norm(subset_points - seed, axis=1).mean()
        shap_values[i] = shap_value(i, feature_count, value_fn)
    return shap_values

if __name__ == "__main__":
    text = "This is an example sentence."
    features = stylometry_features(text)
    print(features)

    points = np.random.rand(10, 96)
    seed = np.random.rand(96)
    voronoi_cell = voronoi_partition(points, seed)
    print(voronoi_cell)

    feature_count = 96
    shap_values = hybrid_stylometric_shapley(text, feature_count)
    print(shap_values)

    shap_values = hybrid_voronoi_shapley(points, seed, feature_count)
    print(shap_values)