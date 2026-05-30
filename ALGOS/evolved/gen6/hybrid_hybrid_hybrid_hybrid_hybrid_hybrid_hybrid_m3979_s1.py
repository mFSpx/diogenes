# DARWIN HAMMER — match 3979, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m2681_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_omni_cha_m701_s1.py (gen4)
# born: 2026-05-29T23:52:54Z

"""
Hybrid Algorithm: Fusing Fisher-based JEPA with RBF Surrogate Learning and NLMS-Adapted Allocation

This module combines the Fisher-based Joint Embedding Predictive Architecture (JEPA) with RBF Surrogate Learning 
from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m2681_s0.py and the NLMS-Adapted Allocation 
from hybrid_hybrid_hybrid_hybrid_hybrid_nlms_omni_cha_m701_s1.py. The mathematical bridge between the two 
structures is the concept of "recovery priority" and the Fisher score, which are used to determine the likelihood 
of an endpoint recovering from a failure and to enhance the encoder output of JEPA. The hybrid operation uses 
the signal and noise scores from the RBF surrogate model as inputs to the NLMS adaptation rule, enabling it to 
learn a mapping between the signal and noise scores and the optimal allocation vector.

The core equations are:

# 1. Fisher score (Parent A)
fisher_score(theta) = exp(-0.5 * (theta - center) / width)

# 2. RBF surrogate model (Parent A)
gaussian(r) = exp(-((epsilon * r) ** 2))

# 3. NLMS adaptation (Parent B)
e      = target – usage
norm_x = x·x + ε
w_new  = w + μ * e * x / norm_x

# 4. Hybrid operation
usage = min( w * M_total , M_available ) 
signal_score = gaussian(euclidean(w, x))
noise_score = 1 - signal_score
w_new  = w + μ * (target - usage) * (fisher_score(theta) * signal_score + noise_score) * x / norm_x
"""

import math
import numpy as np
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def hybrid_operation(w: Vector, x: Vector, M_total: float, M_available: Vector, target: float, 
                     mu: float, epsilon: float, theta: float) -> Vector:
    usage = np.minimum(np.multiply(w, M_total), M_available)
    e = target - np.sum(usage)
    norm_x = np.dot(x, x) + epsilon
    signal_score = gaussian(euclidean(w, x))
    noise_score = 1 - signal_score
    fs = fisher_score(theta)
    w_new = np.add(w, np.multiply(mu * e * (fs * signal_score + noise_score), np.divide(x, norm_x)))
    return w_new

def schedule_allocation(G: int, M_total: float, M_available: Vector, target: float, 
                        mu: float, epsilon: float, theta: float) -> Vector:
    w_base = np.array([math.sin(2 * math.pi * (i / G)) + 1 for i in range(G)])
    w_base = w_base / np.sum(w_base)
    w = hybrid_operation(w_base, w_base, M_total, M_available, target, mu, epsilon, theta)
    return w

def test_hybrid_operation():
    G = 5
    M_total = 100.0
    M_available = np.array([80.0, 90.0, 70.0, 85.0, 95.0])
    target = 400.0
    mu = 0.1
    epsilon = 1e-6
    theta = 0.5

    w = schedule_allocation(G, M_total, M_available, target, mu, epsilon, theta)
    print(w)

if __name__ == "__main__":
    test_hybrid_operation()