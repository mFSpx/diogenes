# DARWIN HAMMER — match 1610, survivor 0
# gen: 4
# parent_a: hybrid_privacy_sketches_m15_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hoeffding_tre_m933_s1.py (gen3)
# born: 2026-05-29T23:37:53Z

"""
Hybrid module that fuses the core mathematics of 
hybrid_privacy_sketches_m15_s3.py and hybrid_hybrid_hybrid_rbf_su_hybrid_hoeffding_tre_m933_s1.py.
The mathematical bridge is the *frequency matrix* `C` from the former 
and the Gaussian radial basis function (RBF) from the latter. 
We use the RBF to model the similarity between quasi-identifier strings 
and inject Laplace noise into the frequency matrix to preserve differential privacy.
"""

import numpy as np
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set
import math

def reconstruction_risk_score(unique_quasi_identifiers: int,
                             total_records: int) -> float:
    """Deterministic risk ratio clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_laplace_noise(scale: float) -> float:
    """Draw a single Laplace(0, scale) sample using numpy."""
    return float(np.random.laplace(0, scale))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def create_frequency_matrix(records: List[Dict[str, Any]], 
                             hash_func: Callable[[str], str], 
                             num_buckets: int, 
                             num_hashes: int) -> np.ndarray:
    """Create a frequency matrix `C` from the given records."""
    frequency_matrix = np.zeros((num_hashes, num_buckets))
    for record in records:
        for hash_index in range(num_hashes):
            hashed_value = int(hash_func(str(record).encode('utf-8')).hexdigest(), 16)
            bucket_index = hashed_value % num_buckets
            frequency_matrix[hash_index, bucket_index] += 1
    return frequency_matrix

def inject_laplace_noise(frequency_matrix: np.ndarray, 
                         sensitivity: float, 
                         epsilon: float) -> np.ndarray:
    """Inject Laplace noise into the frequency matrix."""
    noisy_frequency_matrix = frequency_matrix.copy()
    for i in range(noisy_frequency_matrix.shape[0]):
        for j in range(noisy_frequency_matrix.shape[1]):
            noisy_frequency_matrix[i, j] += dp_laplace_noise(sensitivity / epsilon)
    return noisy_frequency_matrix

def estimate_unique_quasi_identifiers(noisy_frequency_matrix: np.ndarray) -> int:
    """Estimate the number of unique quasi-identifiers from the noisy frequency matrix."""
    return len([i for i in range(noisy_frequency_matrix.shape[1]) if np.any(noisy_frequency_matrix[:, i] > 0)])

def hybrid_risk_score(records: List[Dict[str, Any]], 
                      hash_func: Callable[[str], str], 
                      num_buckets: int, 
                      num_hashes: int, 
                      sensitivity: float, 
                      epsilon: float) -> float:
    """Hybrid risk score that combines the reconstruction risk score with the RBF."""
    frequency_matrix = create_frequency_matrix(records, hash_func, num_buckets, num_hashes)
    noisy_frequency_matrix = inject_laplace_noise(frequency_matrix, sensitivity, epsilon)
    estimated_unique_quasi_identifiers = estimate_unique_quasi_identifiers(noisy_frequency_matrix)
    return reconstruction_risk_score(estimated_unique_quasi_identifiers, len(records))

def hybrid_similarity_score(records1: List[Dict[str, Any]], 
                             records2: List[Dict[str, Any]], 
                             hash_func: Callable[[str], str], 
                             num_buckets: int, 
                             num_hashes: int, 
                             sensitivity: float, 
                             epsilon: float) -> float:
    """Hybrid similarity score that combines the RBF with the noisy frequency matrix."""
    frequency_matrix1 = create_frequency_matrix(records1, hash_func, num_buckets, num_hashes)
    frequency_matrix2 = create_frequency_matrix(records2, hash_func, num_buckets, num_hashes)
    noisy_frequency_matrix1 = inject_laplace_noise(frequency_matrix1, sensitivity, epsilon)
    noisy_frequency_matrix2 = inject_laplace_noise(frequency_matrix2, sensitivity, epsilon)
    similarity = 0
    for i in range(noisy_frequency_matrix1.shape[0]):
        for j in range(noisy_frequency_matrix1.shape[1]):
            similarity += gaussian(euclidean(noisy_frequency_matrix1[i, j], noisy_frequency_matrix2[i, j]), epsilon)
    return similarity

if __name__ == "__main__":
    records = [{'name': 'John', 'age': 25}, {'name': 'Jane', 'age': 30}, {'name': 'Bob', 'age': 35}]
    hash_func = hashlib.sha256
    num_buckets = 10
    num_hashes = 5
    sensitivity = 1.0
    epsilon = 1.0
    print(hybrid_risk_score(records, hash_func, num_buckets, num_hashes, sensitivity, epsilon))
    print(hybrid_similarity_score(records, records, hash_func, num_buckets, num_hashes, sensitivity, epsilon))