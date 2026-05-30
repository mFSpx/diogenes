# DARWIN HAMMER — match 732, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m379_s0.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py (gen3)
# born: 2026-05-29T23:30:44Z

"""
This module combines the tropical max-plus algebra from hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m379_s0.py
and the radial basis function (RBF) surrogate model from hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py.
The mathematical bridge between these structures lies in the application of the RBF surrogate model 
to predict the tropical polynomial coefficients, which are then used to compute the 
perceptual similarity of state vectors in the tropical max-plus algebra.

The RBF surrogate model is used to predict the output of the tropical polynomial coefficients 
from hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m379_s0.py, which is then used to compute the 
SSIM similarity of state vectors in hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m379_s0.py.

Author: [Your Name]
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return np.sqrt(np.sum((a - b) ** 2))

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def rbf_surrogate(coeffs, x):
    return np.sum(coeffs * np.exp(-((x - coeffs) ** 2) / (2 * 0.1 ** 2)), axis=0)

def hybrid_tropical_rbf(coeffs, x):
    tropical_coeffs = rbf_surrogate(coeffs, x)
    return t_polyval(tropical_coeffs, x)

def compute_ssim(state_vectors):
    ssim_values = []
    for i in range(len(state_vectors)):
        for j in range(i+1, len(state_vectors)):
            vector_i = np.asarray(state_vectors[i])
            vector_j = np.asarray(state_vectors[j])
            rbf_similarity = gaussian(euclidean(vector_i, vector_j))
            ssim_values.append(rbf_similarity * t_polyval([1.0, 2.0, 3.0], vector_i) * t_polyval([1.0, 2.0, 3.0], vector_j))
    return np.mean(ssim_values)

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def should_split(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

if __name__ == "__main__":
    state_vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    print(compute_ssim(state_vectors))
    best_gain = 10.0
    second_best_gain = 5.0
    r = 1.0
    delta = 0.1
    n = 100
    print(should_split(best_gain, second_best_gain, r, delta, n).should_split)