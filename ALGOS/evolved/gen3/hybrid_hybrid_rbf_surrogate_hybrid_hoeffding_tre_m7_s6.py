# DARWIN HAMMER — match 7, survivor 6
# gen: 3
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s0.py (gen2)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# born: 2026-05-29T23:25:20Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple
import numpy as np

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

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def combined_similarity(features: Dict[Node, FeatureVec], epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    K, nodes_k = rbf_kernel_matrix(features, epsilon)
    S, nodes_s = similarity_matrix(features)
    if nodes_k != nodes_s:
        raise ValueError("Node ordering mismatch between K and S")
    C = t_matmul(K, S)
    return C, nodes_k

def node_gain_matrix(combined: np.ndarray) -> np.ndarray:
    return np.max(combined, axis=1)

def hybrid_split_decision(gains: np.ndarray, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    if gains.size < 2:
        return SplitDecision(True, 0.0, 0.0, "single_candidate")
    sorted_idx = np.argsort(gains)[::-1]
    best = gains[sorted_idx[0]]
    second = gains[sorted_idx[1]]
    return should_split(best, second, r, delta, n, tie_threshold)

def modulated_probability(raw_p: float, node_idx: int, undecided_mask: np.ndarray, combined: np.ndarray) -> float:
    if not (0.0 <= raw_p <= 1.0):
        raise ValueError("raw_p must be in [0,1]")
    row = combined[node_idx]
    relevant = row[undecided_mask]
    if relevant.size == 0:
        return raw_p
    modulation = np.mean(relevant)
    return raw_p * modulation

def improved_combined_similarity(features: Dict[Node, FeatureVec], epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    K, nodes_k = rbf_kernel_matrix(features, epsilon)
    S, nodes_s = similarity_matrix(features)
    if nodes_k != nodes_s:
        raise ValueError("Node ordering mismatch between K and S")
    C = t_matmul(K, np.sqrt(S))  # Improved fusion using square root
    return C, nodes_k

def improved_hybrid_split_decision(gains: np.ndarray, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    if gains.size < 2:
        return SplitDecision(True, 0.0, 0.0, "single_candidate")
    sorted_idx = np.argsort(gains)[::-1]
    best = gains[sorted_idx[0]]
    second = gains[sorted_idx[1]]
    eps = hoeffding_bound(r, delta, n)
    gap = best - second
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def improved_modulated_probability(raw_p: float, node_idx: int, undecided_mask: np.ndarray, combined: np.ndarray) -> float:
    if not (0.0 <= raw_p <= 1.0):
        raise ValueError("raw_p must be in [0,1]")
    row = combined[node_idx]
    relevant = row[undecided_mask]
    if relevant.size == 0:
        return raw_p
    modulation = np.mean(np.sqrt(relevant))  # Improved modulation using square root
    return raw_p * modulation