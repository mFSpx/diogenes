# DARWIN HAMMER — match 2789, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_semant_m2556_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s2.py (gen5)
# born: 2026-05-29T23:45:53Z

"""
This module fuses the governing equations of hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_semant_m2556_s1.py 
and hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s2.py. 
The mathematical bridge between their structures lies in the use of Gaussian functions 
and similarity matrices. Specifically, the sphericity and flatness indices from the first 
algorithm are used to compute a weighted similarity matrix, which is then used as input 
to the RBF kernel matrix computation from the second algorithm.

Parents: 
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_semant_m2556_s1.py
- hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s2.py
"""

import numpy as np
import math
from typing import List, Tuple

Point = Tuple[float, float]
Node = str
Graph = dict
FeatureVec = Tuple[float, float, float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def hybrid_entropy(length: float, width: float, height: float) -> float:
    sphericity = sphericity_index(length, width, height)
    flatness = flatness_index(length, width, height)
    return -sphericity * math.log(max(sphericity, 1e-12)) - flatness * math.log(max(flatness, 1e-12))

def compute_similarity_matrix(morphologies: List[Tuple[float, float, float]], 
                              epsilon: float = 1.0) -> np.ndarray:
    n = len(morphologies)
    S = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            length_i, width_i, height_i = morphologies[i]
            length_j, width_j, height_j = morphologies[j]
            sphericity_i = sphericity_index(length_i, width_i, height_i)
            flatness_i = flatness_index(length_i, width_i, height_i)
            sphericity_j = sphericity_index(length_j, width_j, height_j)
            flatness_j = flatness_index(length_j, width_j, height_j)
            features_i: FeatureVec = (sphericity_i, flatness_i, hybrid_entropy(length_i, width_i, height_i))
            features_j: FeatureVec = (sphericity_j, flatness_j, hybrid_entropy(length_j, width_j, height_j))
            dist = euclidean(features_i, features_j)
            sim = gaussian(dist, epsilon)
            S[i, j] = sim
            S[j, i] = sim
    return S

def rbf_kernel_matrix(similarity_matrix: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    n = similarity_matrix.shape[0]
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = 1 - similarity_matrix[i, j]
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K

def hybrid_operation(morphologies: List[Tuple[float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    similarity_matrix = compute_similarity_matrix(morphologies)
    rbf_kernel_matrix_result = rbf_kernel_matrix(similarity_matrix)
    return similarity_matrix, rbf_kernel_matrix_result

if __name__ == "__main__":
    morphologies = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    similarity_matrix, rbf_kernel_matrix_result = hybrid_operation(morphologies)
    print(similarity_matrix)
    print(rbf_kernel_matrix_result)