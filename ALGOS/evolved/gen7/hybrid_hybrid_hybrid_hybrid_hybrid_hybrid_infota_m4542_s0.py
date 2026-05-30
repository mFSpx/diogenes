# DARWIN HAMMER — match 4542, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0.py (gen6)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.py (gen3)
# born: 2026-05-29T23:56:31Z

"""
Hybrid Algorithm - fusion of hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m2240_s0 and hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s1.

The mathematical bridge between the two parents lies in using the Fisher-information score from the first parent as the uncertainty
for the burst admission score calculation in the Chelydrid Ambush-Strike model from the second parent. This allows us to estimate 
the similarity between probability distributions using approximate Jaccard similarity and determine whether to select an element 
as the representative of a cluster based on the similarity between clusters. The RBF kernel matrix from the first parent is used 
to compute the Fisher-information score, which is then used to weight the perceptual similarity scores.

"""

import math
import random
import numpy as np
import sys
import pathlib

Node = object
FeatureVec = list[float]

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def compute_phash(values: list[float]) -> int:
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
    features: dict[Node, FeatureVec], epsilon: float = 1.0
) -> tuple[np.ndarray, list[Node]]:
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
    # simple implementation of MinHash
    signature = []
    for i in range(k):
        hash_value = hash(str([p * i for p in probabilities]))
        signature.append(hash_value)
    return signature

def compute_fisher_information(K: np.ndarray, c: float, w: float) -> float:
    """
    Compute the Fisher-information score I(d; c, w) with a centre c and width w.
    """
    n = K.shape[0]
    I = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.sqrt((K[i, i] - K[j, j]) ** 2)
                I += (dist ** 2) * gaussian(dist, w)
    I /= (c ** 2) * (w ** 2)
    return I

def hybrid_similarity_matrix(features: dict[Node, FeatureVec], epsilon: float = 1.0) -> np.ndarray:
    """
    Compute the hybrid similarity matrix using the RBF kernel matrix and the Fisher-information score.
    """
    K, nodes = rbf_kernel_matrix(features, epsilon)
    c = np.mean(K)
    w = np.std(K)
    I = compute_fisher_information(K, c, w)
    similarity_matrix = np.zeros_like(K)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            similarity_matrix[i, j] = I * gaussian(euclidean(features[nodes[i]], features[nodes[j]]), epsilon)
    return similarity_matrix

if __name__ == "__main__":
    features = {
        'node1': [1.0, 2.0, 3.0],
        'node2': [4.0, 5.0, 6.0],
        'node3': [7.0, 8.0, 9.0]
    }
    similarity_matrix = hybrid_similarity_matrix(features)
    print(similarity_matrix)