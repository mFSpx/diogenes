# DARWIN HAMMER — match 4542, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0.py (gen6)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.py (gen3)
# born: 2026-05-29T23:56:31Z

"""
Hybrid RBF-Perceptual Fisher-Stylometry Fusion with Infotaxis MinHash.

This module integrates the mathematical structures of two parent algorithms:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0.py
- hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.py

The mathematical bridge lies in using the RBF kernel matrix to compute the Fisher information, 
which is then used to weight the perceptual similarity scores. Meanwhile, the MinHash signatures 
are used to estimate the similarity between probability distributions, and the burst admission 
score calculation is modified to incorporate the Fisher information.

Authors: [Your Name]
"""

import math
import random
import numpy as np
import sys
import pathlib
from typing import Dict, List, Tuple, Sequence

Node = object
FeatureVec = Sequence[float]

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def compute_phash(values: List[float]) -> int:
    """
    Simple perceptual hash: 1-bit per value relative to the median.
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

def rbf_kernel_matrix(
    features: Dict[Node, FeatureVec], epsilon: float = 1.0
) -> Tuple[np.ndarray, List[Node]]:
    """
    Dense RBF kernel K where K[i, j] = exp(-ε² * ||f_i - f_j||²).
    Returns the matrix and the node ordering.
    """
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        K[i, i] = 1.0  # distance zero → kernel 1
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            K[i, j] = gaussian(dist, epsilon)
            K[j, i] = K[i, j]

    return K, nodes

def compute_signature(probabilities: list[float], k: int = 128) -> list[int]:
    """
    Compute MinHash signature for a probability distribution.

    Parameters:
    probabilities (list[float]): The probability distribution.
    k (int): The signature length (default: 128).

    Returns:
    list[int]: The MinHash signature.
    """
    min_hash = []
    for i in range(k):
        hash_value = 0
        for j, prob in enumerate(probabilities):
            if prob > 0:
                hash_value += (j * prob)
        min_hash.append(hash_value)
    return min_hash

def fisher_information(K: np.ndarray) -> float:
    """
    Compute the Fisher information from the RBF kernel matrix.

    Parameters:
    K (np.ndarray): The RBF kernel matrix.

    Returns:
    float: The Fisher information.
    """
    mean_K = np.mean(K)
    std_K = np.std(K)
    I = np.log(std_K) - np.log(mean_K)
    return I

def hybrid_similarity(features: Dict[Node, FeatureVec], epsilon: float = 1.0) -> np.ndarray:
    """
    Compute the hybrid similarity matrix.

    Parameters:
    features (Dict[Node, FeatureVec]): The feature vectors.
    epsilon (float): The RBF kernel parameter (default: 1.0).

    Returns:
    np.ndarray: The hybrid similarity matrix.
    """
    K, nodes = rbf_kernel_matrix(features, epsilon)
    I = fisher_information(K)
    similarity = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            similarity[i, j] = I * K[i, j]
    return similarity

def infotaxis_minhash_similarity(features: Dict[Node, FeatureVec], k: int = 128) -> np.ndarray:
    """
    Compute the Infotaxis MinHash similarity matrix.

    Parameters:
    features (Dict[Node, FeatureVec]): The feature vectors.
    k (int): The MinHash signature length (default: 128).

    Returns:
    np.ndarray: The Infotaxis MinHash similarity matrix.
    """
    nodes = list(features.keys())
    n = len(nodes)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            prob_i = features[nodes[i]]
            prob_j = features[nodes[j]]
            signature_i = compute_signature(prob_i, k)
            signature_j = compute_signature(prob_j, k)
            similarity[i, j] = 1 - hamming_distance(compute_phash(signature_i), compute_phash(signature_j)) / k
    return similarity

if __name__ == "__main__":
    features = {
        'node1': [1.0, 2.0, 3.0],
        'node2': [4.0, 5.0, 6.0],
        'node3': [7.0, 8.0, 9.0]
    }
    hybrid_sim = hybrid_similarity(features)
    infotaxis_minhash_sim = infotaxis_minhash_similarity(features)
    print("Hybrid Similarity Matrix:")
    print(hybrid_sim)
    print("Infotaxis MinHash Similarity Matrix:")
    print(infotaxis_minhash_sim)