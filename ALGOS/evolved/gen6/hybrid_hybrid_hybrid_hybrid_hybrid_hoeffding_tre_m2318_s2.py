# DARWIN HAMMER — match 2318, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_model__m860_s3.py (gen3)
# parent_b: hybrid_hoeffding_tree_hybrid_gini_coeffici_m685_s2.py (gen5)
# born: 2026-05-29T23:41:53Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Tuple, Iterable
import numpy as np

def tropical_distance(x: np.ndarray, y: np.ndarray) -> float:
    return np.max(np.abs(x - y))

def allocate_features(num_nodes: int, feature_dim: int, budget_mb: int = 4096) -> np.ndarray:
    max_bytes = budget_mb * 1024 * 1024
    required_bytes = num_nodes * feature_dim * 4
    if required_bytes > max_bytes:
        feature_dim = max_bytes // (num_nodes * 4)
    return np.random.uniform(size=(num_nodes, feature_dim))

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def modulated_hoeffding_bound(r: float, delta: float, n: int, gini: float) -> float:
    epsilon = 0.1
    gamma = math.exp(-(epsilon * gini) ** 2)
    return hoeffding_bound(r, delta, n) * gamma

def hybrid_split_decision(node_features: np.ndarray, delta: float, n: int) -> SplitDecision:
    distances = np.zeros((node_features.shape[0], node_features.shape[0]))
    for i in range(node_features.shape[0]):
        for j in range(i+1, node_features.shape[0]):
            distances[i, j] = tropical_distance(node_features[i], node_features[j])
            distances[j, i] = distances[i, j]

    gini_coefficients = np.array([gini_coefficient(node_features[i]) for i in range(node_features.shape[0])])

    modulated_bounds = np.array([modulated_hoeffding_bound(np.max(distances[i]), delta, n, gini_coefficients[i]) for i in range(node_features.shape[0])])

    should_split = np.any(modulated_bounds > 0)
    epsilon = np.max(modulated_bounds)
    gain_gap = np.max(distances) - np.min(distances)
    reason = "Split" if should_split else "Don't split"
    return SplitDecision(should_split, epsilon, gain_gap, reason)

def improved_hybrid_split_decision(node_features: np.ndarray, delta: float, n: int) -> SplitDecision:
    node_centroids = np.mean(node_features, axis=0)
    distances_to_centroid = np.array([tropical_distance(node_features[i], node_centroids) for i in range(node_features.shape[0])])
    gini_coefficients = np.array([gini_coefficient(node_features[i]) for i in range(node_features.shape[0])])

    modulated_bounds = np.array([modulated_hoeffding_bound(distances_to_centroid[i], delta, n, gini_coefficients[i]) for i in range(node_features.shape[0])])

    should_split = np.any(modulated_bounds > 0)
    epsilon = np.max(modulated_bounds)
    gain_gap = np.max(distances_to_centroid) - np.min(distances_to_centroid)
    reason = "Split" if should_split else "Don't split"
    return SplitDecision(should_split, epsilon, gain_gap, reason)

if __name__ == "__main__":
    node_features = allocate_features(10, 5)
    delta = 0.1
    n = 100
    decision = improved_hybrid_split_decision(node_features, delta, n)
    print(decision)