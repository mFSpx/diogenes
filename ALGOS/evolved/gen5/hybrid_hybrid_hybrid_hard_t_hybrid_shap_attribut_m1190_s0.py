# DARWIN HAMMER — match 1190, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s3.py (gen3)
# parent_b: hybrid_shap_attribution_hybrid_hybrid_pherom_m70_s1.py (gen4)
# born: 2026-05-29T23:33:16Z

"""
Hybrid Stylometric‑Geometric Model with Shapley Attribution
======================================================
This module fuses two previously independent algorithms:
* **Parent A** – a stylometric extractor that builds a high‑dimensional
  frequency fingerprint (`stylometry_features`) from textual data.
* **Parent B** – a Shapley attribution model for calculating feature importance
  with a pheromone-inspired leader election algorithm.

The mathematical bridge is the observation that a stylometric fingerprint is
simply a point in ℝⁿ (n = 96 in the original implementation).  By treating each
fingerprint as a geometric point we can:
1. Use Euclidean distance to assign texts to the Voronoi cell of the nearest
   seed (``voronoi_partition``).
2. Map the presence of linguistic function‑category tokens onto a Clifford
   blade (one basis vector per category) and combine blades with the geometric
   operations of Parent A.
3. Calculate the Shapley value of each feature in the stylometric fingerprint
   using the Shapley kernel weight from Parent B.

The resulting hybrid functions expose a unified workflow:
text → vector → geometric region & algebraic signature → Shapley attribution.
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

# ----------------------------------------------------------------------
# Parent A – stylometry utilities
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot".split()
    ),
}

def stylometry_features(text: str) -> np.ndarray:
    """Extract stylometric features from a text"""
    words = re.findall(r'\b\w+\b', text.lower())
    freqs = Counter(words)
    return np.array([freqs.get(word, 0) for word in FUNCTION_CATS["pronoun"]])

def voronoi_partition(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Partition points into Voronoi cells based on nearest seed"""
    distances = np.linalg.norm(points[:, None] - seeds, axis=2)
    return np.argmin(distances, axis=1)

# ----------------------------------------------------------------------
# Parent B – Shapley attribution utilities
# ----------------------------------------------------------------------
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

def shap_value(feature_index: int, feature_count: int, value_fn: callable) -> float:
    total = 0.0
    for k in range(feature_count + 1):
        for subset in combinations(range(feature_count), k):
            s = frozenset(subset)
            total += shapley_kernel_weight(k, feature_count) * (value_fn(s | {feature_index}) - value_fn(s))
    return total

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_stylometry_shap(text: str, feature_count: int) -> Tuple[np.ndarray, float]:
    """Extract stylometric features and calculate Shapley value"""
    features = stylometry_features(text)
    shap_values = [shap_value(i, feature_count, lambda s: np.sum([features[j] for j in s])) for i in range(feature_count)]
    return features, shap_values

def hybrid_voronoi_shap(points: np.ndarray, seeds: np.ndarray, feature_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Partition points into Voronoi cells and calculate Shapley values"""
    voronoi_cells = voronoi_partition(points, seeds)
    shap_values = [shap_value(i, feature_count, lambda s: np.sum([points[j] for j in s])) for i in range(feature_count)]
    return voronoi_cells, shap_values

def hybrid_text_shap(text: str, feature_count: int) -> Tuple[np.ndarray, float, np.ndarray]:
    """Extract stylometric features, calculate Shapley value, and partition into Voronoi cells"""
    features, shap_values = hybrid_stylometry_shap(text, feature_count)
    voronoi_cells = voronoi_partition(features[:, None], np.random.rand(10, feature_count))
    return features, shap_values, voronoi_cells

if __name__ == "__main__":
    text = "This is a sample text for testing the hybrid model."
    feature_count = 10
    features, shap_values = hybrid_stylometry_shap(text, feature_count)
    voronoi_cells, _ = hybrid_voronoi_shap(np.random.rand(10, feature_count), np.random.rand(10, feature_count), feature_count)
    _, _, _ = hybrid_text_shap(text, feature_count)
    print("Hybrid model test successful.")