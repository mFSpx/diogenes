# DARWIN HAMMER — match 125, survivor 0
# gen: 4
# parent_a: hybrid_infotaxis_minhash_m63_s3.py (gen1)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s7.py (gen3)
# born: 2026-05-29T23:26:54Z

"""
HYBRID ALGORITHM: fusion of infotaxis_minhash and hybrid_rbf_surrogate_hoeffding_tree

This module integrates the core topologies of infotaxis_minhash (A) and
hybrid_rbf_surrogate_hoeffding_tree (B) into a single unified system.

Parent A provides a MinHash signature-based similarity metric,
while Parent B leverages a Gaussian RBF kernel for feature similarity
and a Hoeffding tree for decision-making. The mathematical bridge between
their structures lies in the use of hash-based signatures and Gaussian
kernels, which can be combined to produce a probabilistic, information-theoretic
representation of similarity.

In this fusion, we use the MinHash signatures to compute a probability-like
representation of similarity between feature vectors, and then apply the
Gaussian RBF kernel to compute the expected entropy of each action given the
feature similarity. The resulting expected entropy is used to select the best
action.
"""

import numpy as np
import math
import random
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Set, Tuple, Dict

MAX64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# MinHash core
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Entropy core
# ----------------------------------------------------------------------
def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete distribution given by *probabilities*."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted average entropy of two possible states."""
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def best_action(actions: Dict[str, Tuple[float, List[int], List[int]]]) -> str:
    """Select the action with minimal expected entropy."""
    if not actions:
        raise ValueError("actions must not be empty")


# ----------------------------------------------------------------------
# RBF kernel & perceptual similarity
# ----------------------------------------------------------------------
def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def compute_phash(values: List[float]) -> int:
    """
    Simple perceptual hash: 1‑bit per value relative to the median.
    Uses up to 64 bits; remaining values are ignored.
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit strings."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def rbf_minhash_kernel(
    features: Dict[str, List[float]], k: int = 128
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute RBF kernel matrix on MinHash signatures.
    """
    signatures = {key: signature(values) for key, values in features.items()}
    K, keys = rbf_kernel_matrix(signatures, epsilon=1.0)
    return K, keys


def hybrid_similarity_matrix(
    features: Dict[str, List[float]], k: int = 128
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute hybrid similarity matrix using RBF kernel on MinHash signatures.
    """
    K, keys = rbf_minhash_kernel(features, k=k)
    return K, keys


def expected_entropy_hybrid(
    features: Dict[str, List[float]], k: int = 128
) -> float:
    """
    Compute expected entropy of hybrid similarity matrix.
    """
    K, keys = hybrid_similarity_matrix(features, k=k)
    if not K.size:
        return 0.0
    entropy_values = []
    for i in range(K.shape[0]):
        row = K[i, :]
        entropy_values.append(expected_entropy(1.0, row, [1.0 - x for x in row]))
    return np.mean(entropy_values)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    features = {
        "key1": [1.0, 2.0, 3.0],
        "key2": [4.0, 5.0, 6.0],
        "key3": [7.0, 8.0, 9.0],
    }
    k = 128
    K, keys = hybrid_similarity_matrix(features, k=k)
    expected_entropy_value = expected_entropy_hybrid(features, k=k)
    print("Expected Entropy:", expected_entropy_value)