# DARWIN HAMMER — match 223, survivor 2
# gen: 4
# parent_a: nlms.py (gen0)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s5.py (gen3)
# born: 2026-05-29T23:27:36Z

import math
import numpy as np
from dataclasses import dataclass
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

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[n])) for n in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
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
                 tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_best_gain
    if gain_gap < tie_threshold * max(best_gain, second_best_gain):
        reason = "Tie threshold exceeded"
    elif gain_gap <= eps:
        reason = "Insufficient data to make a decision"
    else:
        reason = "Gain gap is significant"
    should_split = True if gain_gap > eps else False
    return SplitDecision(should_split, eps, gain_gap, reason)

def predict(weights: Iterable[float], x: Iterable[float]) -> float:
    return sum(w * xi for w, xi in zip(weights, x))

def update(weights: list[float], x: list[float], target: float, mu: float = 0.5, eps: float = 1e-9) -> tuple[list[float], float]:
    if len(weights) != len(x):
        raise ValueError('weights and input must have equal length')
    if not 0 < mu < 2:
        raise ValueError('mu must be in the interval (0, 2)')
    y = predict(weights, x)
    error = target - y
    power = sum(xi * xi for xi in x) + eps
    next_weights = [w + mu * error * xi / power for w, xi in zip(weights, x)]
    return next_weights, error

def hybrid_update(weights: list[float], features: Dict[Node, FeatureVec], epsilon: float = 1.0, mu: float = 0.5, eps: float = 1e-9) -> tuple[list[float], float]:
    K, nodes = rbf_kernel_matrix(features, epsilon)
    S, nodes = similarity_matrix(features)
    n = len(nodes)
    K_inv = np.linalg.inv(K)
    S_inv = np.linalg.inv(S)

    # Compute the weighted sum of the features
    weighted_features = np.dot(K_inv, features[nodes[0]])
    weighted_similarity = np.dot(K_inv, S)
    next_weights = []
    for i in range(n):
        x = weighted_features[i]
        S_i = weighted_similarity[i]
        y = predict(weights, x)
        error = 1.0 - y
        power = sum(xi * xi for xi in x) + eps
        next_weights.append(w + mu * error * xi / power for w, xi in zip(weights, x))

    # Update the weights using the similarity matrix
    updated_weights = np.dot(S_inv, next_weights)
    return updated_weights.tolist(), error

def hybrid_hoeffding_bound(r: float, delta: float, n: int) -> float:
    return hoeffding_bound(r, delta, n)

def hybrid_should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gain_gap = best_gain - second_best_gain
    if gain_gap < tie_threshold * max(best_gain, second_best_gain):
        reason = "Tie threshold exceeded"
    elif gain_gap <= eps:
        reason = "Insufficient data to make a decision"
    else:
        reason = "Gain gap is significant"
    should_split = True if gain_gap > eps else False
    return SplitDecision(should_split, eps, gain_gap, reason)

if __name__ == "__main__":
    # Smoke test
    weights = [1.0, 2.0, 3.0]
    features = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
    epsilon = 1.0
    mu = 0.5
    eps = 1e-9
    updated_weights, error = hybrid_update(weights, features, epsilon, mu, eps)
    print(updated_weights)
    print(error)