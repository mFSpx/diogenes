# DARWIN HAMMER — match 978, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_model_vram_sc_m61_s0.py (gen4)
# parent_b: hybrid_nlms_hybrid_hybrid_rbf_su_m223_s2.py (gen4)
# born: 2026-05-29T23:32:08Z

import numpy as np
import math
import random
from pathlib import Path
from typing import Hashable, Sequence, List, Dict, Set, Tuple

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec], vram_budget_mb: int) -> Tuple[np.ndarray, List[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    epsilon = 1.0 / (vram_budget_mb / 1024.0)  
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[n])) for n in nodes]

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            sim = gaussian(dist, epsilon)
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def rbf_kernel_matrix(features: Dict[Node, FeatureVec], epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> bool:
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_best_gain
    if gain_gap < tie_threshold * max(best_gain, second_best_gain):
        return False
    return gain_gap > eps

def hybrid_rbf_similarity(features: Dict[Node, FeatureVec], vram_budget_mb: int) -> Tuple[np.ndarray, List[Node]]:
    S, nodes = similarity_matrix(features, vram_budget_mb)
    K, _ = rbf_kernel_matrix(features)
    combined_matrix = np.multiply(S, K)
    return combined_matrix, nodes

def hybrid_hoeffding_tree(features: Dict[Node, FeatureVec], vram_budget_mb: int) -> bool:
    combined_matrix, _ = hybrid_rbf_similarity(features, vram_budget_mb)
    best_gain = np.max(combined_matrix)
    second_best_gain = np.partition(combined_matrix.flatten(), -2)[-2]
    return should_split(best_gain, second_best_gain, 0.1, 0.05, 100)

def adaptive_epsilon(vram_budget_mb: int, n: int) -> float:
    return 1.0 / (vram_budget_mb / 1024.0 * math.log(n))

def improved_hybrid_rbf_similarity(features: Dict[Node, FeatureVec], vram_budget_mb: int) -> Tuple[np.ndarray, List[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    epsilon = adaptive_epsilon(vram_budget_mb, n)
    S, _ = similarity_matrix(features, vram_budget_mb)
    K, _ = rbf_kernel_matrix(features, epsilon)
    combined_matrix = np.multiply(S, K)
    return combined_matrix, nodes

def improved_hybrid_hoeffding_tree(features: Dict[Node, FeatureVec], vram_budget_mb: int) -> bool:
    combined_matrix, _ = improved_hybrid_rbf_similarity(features, vram_budget_mb)
    best_gain = np.max(combined_matrix)
    second_best_gain = np.partition(combined_matrix.flatten(), -2)[-2]
    return should_split(best_gain, second_best_gain, 0.1, 0.05, 100)

if __name__ == "__main__":
    features = {i: [random.random() for _ in range(10)] for i in range(10)}
    vram_budget_mb = 1024
    result = improved_hybrid_hoeffding_tree(features, vram_budget_mb)
    print(result)