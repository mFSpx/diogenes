# DARWIN HAMMER — match 4716, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m717_s3.py (gen4)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s3.py (gen5)
# born: 2026-05-29T23:57:36Z

"""
This module fuses the mathematical structures of two parent algorithms:
- `hybrid_hybrid_hybrid_bandit_label_foundry_m21_s3.py` (DARWIN HAMMER — match 717, survivor 3)
- `hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s3.py` (DARWIN HAMMER — match 1061, survivor 3)

The fusion integrates the Count-Min Sketch (CMS) and HyperLogLog (HLL) data structures from the first parent
with the Gaussian-based similarity metrics from the second parent. The mathematical interface is established
through the shared use of hash functions and vector comparisons.

Specifically, the CMS and HLL structures are used to estimate the cardinality of sets, while the Gaussian-based
similarity metrics are used to compute similarity scores between vectors. The hybrid algorithm combines these
components to produce a novel similarity metric that integrates both data structures.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, Dict

class CountMinSketch:
    """Count-Min sketch for non-negative integer streams."""

    def __init__(self, width: int = 2000, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2 ** 31 - 1) for _ in range(depth)]

    def _hash(self, key: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=8, key=self.seeds[i].to_bytes(4, "little"))
        h.update(key.encode("utf-8"))
        return int.from_bytes(h.digest(), "big") % self.width

    def update(self, key: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += increment

    def estimate(self, key: str) -> int:
        """Return the minimum count over all hash rows (the CMS estimate)."""
        mins = [self.tables[i, self._hash(key, i)] for i in range(self.depth)]
        return min(mins)

    def total(self) -> int:
        """Total count of all updates (sum over first row, approximates N)."""
        return int(self.tables.sum())


class HyperLogLog:
    """Very small HyperLogLog implementation for cardinality estimation."""

    def __init__(self, b: int = 10):
        self.b = b  # number of bits for register index
        self.m = 1 << b
        self.registers = np.zeros(self.m, dtype=np.uint8)

    def _rho(self, w: int) -> int:
        """Position of first 1-bit in w (1-based)."""
        return (w & -w).bit_length()

    def add(self, item: str) -> None:
        h = int(hashlib.sha1(item.encode("utf-8")).hexdigest(), 16)
        idx = h >> (64 - self.b)
        w = (h << self.b) & ((1 << 64) - 1)
        rank = self._rho(w)
        self.registers[idx] = max(self.registers[idx], rank)

    def cardinality(self) -> float:
        """Return the cardinality estimate."""
        return 1.0 / (self.m * math.pow(2, -self.registers.max()))


class HybridSimilarity:
    """Hybrid similarity metric combining Count-Min Sketch and HyperLogLog with Gaussian-based similarity metrics."""

    def __init__(self, cms_width: int = 2000, cms_depth: int = 5, hll_b: int = 10, epsilon: float = 1.0):
        self.cms = CountMinSketch(cms_width, cms_depth)
        self.hll = HyperLogLog(hll_b)
        self.epsilon = epsilon

    def add(self, key: str) -> None:
        self.cms.update(key)
        self.hll.add(key)

    def estimate(self, key: str) -> float:
        """Return the hybrid similarity score."""
        cms_estimate = self.cms.estimate(key)
        hll_estimate = self.hll.cardinality()
        dist = euclidean([cms_estimate, hll_estimate], [0.5, 0.5])
        return gaussian(dist, self.epsilon)

    def total(self) -> float:
        """Total count of all updates (sum over first row, approximates N)."""
        return self.cms.total()


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
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


def similarity_matrix(features: Dict[int, List[float]]) -> Tuple[np.ndarray, List[int]]:
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


def rbf_kernel_matrix(features: Dict[int, List[float]], epsilon: float = 1.0) -> Tuple[np.ndarray, List[int]]:
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


def lsm_score(v1: List[float], v2: List[float]) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def hybrid_similarity_test():
    features = {1: [1.0, 2.0, 3.0], 2: [4.0, 5.0, 6.0]}
    hybrid = HybridSimilarity()
    for k in features.keys():
        hybrid.add(str(k))
    S, _ = similarity_matrix(features)
    K, _ = rbf_kernel_matrix(features)
    print(hybrid.estimate("1"))
    print(np.exp(-((1.0 - 0.5) ** 2)))
    print(S[0, 0])
    print(K[0, 0])


if __name__ == "__main__":
    hybrid_similarity_test()