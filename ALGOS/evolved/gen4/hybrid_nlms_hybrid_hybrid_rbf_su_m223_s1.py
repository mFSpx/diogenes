# DARWIN HAMMER — match 223, survivor 1
# gen: 4
# parent_a: nlms.py (gen0)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s5.py (gen3)
# born: 2026-05-29T23:27:36Z

"""
Module for the Hybrid NLMS-RBF algorithm.

This module combines the Normalized Least Mean Squares (NLMS) update rule from nlms.py with the 
Radial Basis Function (RBF) kernel matrix computation from hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s5.py.
The mathematical bridge between the two lies in the use of the RBF kernel matrix to transform the input data,
which is then used in the NLMS update rule to update the weights.

The idea is to use the RBF kernel matrix to project the input data into a higher-dimensional space,
where the NLMS update rule can be applied to update the weights. This allows the algorithm to capture more complex relationships
between the input data and the target values.
"""

import numpy as np
import math
import random
import sys
import pathlib

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

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

def similarity_matrix(features: dict[int, list[float]]) -> tuple[np.ndarray, list[int]]:
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

def rbf_kernel_matrix(features: dict[int, list[float]], epsilon: float = 1.0) -> tuple[np.ndarray, list[int]]:
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

def predict(weights: list[float], x: list[float], K: np.ndarray) -> float:
    return sum(w * K[i, i] * xi for i, (w, xi) in enumerate(zip(weights, x)))

def update(weights: list[float], x: list[float], target: float, mu: float = 0.5, eps: float = 1e-9, K: np.ndarray = None) -> tuple[list[float], float]:
    if K is None:
        raise ValueError("RBF kernel matrix is required")
    if len(weights) != len(x):
        raise ValueError('weights and input must have equal length')
    if not 0 < mu < 2:
        raise ValueError('mu must be in the interval (0, 2)')
    y = predict(weights, x, K)
    error = target - y
    power = sum(K[i, i] * xi * xi for i, xi in enumerate(x)) + eps
    next_weights = [w + mu * error * xi * K[i, i] / power for i, (w, xi) in enumerate(zip(weights, x))]
    return next_weights, error

def hybrid_nlms_rbf(x: list[float], target: float, mu: float = 0.5, eps: float = 1e-9, epsilon: float = 1.0) -> tuple[list[float], float, np.ndarray]:
    features = {i: [xi] for i, xi in enumerate(x)}
    K, _ = rbf_kernel_matrix(features, epsilon)
    weights = [1.0 / len(x) for _ in x]
    next_weights, error = update(weights, x, target, mu, eps, K)
    return next_weights, error, K

if __name__ == "__main__":
    x = [1.0, 2.0, 3.0]
    target = 4.0
    mu = 0.5
    eps = 1e-9
    epsilon = 1.0
    next_weights, error, K = hybrid_nlms_rbf(x, target, mu, eps, epsilon)
    print(f"Next weights: {next_weights}")
    print(f"Error: {error}")
    print(f"RBF kernel matrix: {K}")