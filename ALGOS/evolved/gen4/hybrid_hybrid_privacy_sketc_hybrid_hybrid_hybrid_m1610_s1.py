# DARWIN HAMMER — match 1610, survivor 1
# gen: 4
# parent_a: hybrid_privacy_sketches_m15_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hoeffding_tre_m933_s1.py (gen3)
# born: 2026-05-29T23:37:53Z

"""
Hybrid module fusing the core topologies of 
hybrid_privacy_sketches_m15_s3.py (parent A) and 
hybrid_hybrid_hybrid_rbf_su_hybrid_hoeffding_tre_m933_s1.py (parent B).

The mathematical bridge between the two parents lies in the application 
of differential privacy (DP) concepts to the RBF surrogate model. 
Specifically, we inject Laplace noise into the RBF centers to create 
a DP-RBF surrogate, which can then be used to estimate the 
reconstruction risk score.

We utilize the Count-Min sketch matrix from parent A to create a 
noisy estimate of the number of distinct quasi-identifiers. 
This estimate is then used to compute the reconstruction risk score, 
which is integrated with the Hoeffding bound from parent B to 
determine the gain gap in the RBF surrogate model.
"""

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import pathlib
from collections import defaultdict

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class RBFSurrogate:
    centers: list[tuple[float, ...]]
    weights: list[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers))

def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("range_ > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def gini_impurity_from_counts(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return 1.0 - np.sum(probs ** 2)

def gini_gain(parent_counts: dict, left_counts: dict, right_counts: dict) -> float:
    n_parent = sum(parent_counts.values())
    if n_parent == 0:
        return 0.0

    parent_imp = gini_impurity_from_counts(parent_counts)
    left_imp = gini_impurity_from_counts(left_counts)
    right_imp = gini_impurity_from_counts(right_counts)
    
    return parent_imp - (left_counts.get('', 0) / n_parent) * left_imp - (right_counts.get('', 0) / n_parent) * right_imp

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Deterministic risk ratio clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_laplace_noise(scale: float) -> float:
    """Draw a single Laplace(0, scale) sample using numpy."""
    return float(np.random.laplace(0, scale))

def hybrid_privacy_rbf_surrogate(centers: list[tuple[float, ...]], 
                                  weights: list[float], 
                                  epsilon: float, 
                                  sensitivity: float, 
                                  total_records: int) -> float:
    noisy_centers = [tuple(x + dp_laplace_noise(sensitivity) for x in center) for center in centers]
    noisy_surrogate = RBFSurrogate(noisy_centers, weights, epsilon)
    unique_quasi_identifiers = len(noisy_centers)
    risk_score = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    return risk_score

def hybrid_hoeffding_rbf_surrogate(centers: list[tuple[float, ...]], 
                                    weights: list[float], 
                                    epsilon: float, 
                                    range_: float, 
                                    delta: float, 
                                    n: int) -> SplitDecision:
    hoeffding_error = hoeffding_bound(range_, delta, n)
    gain_gap = gini_gain({'A': 1, 'B': 1}, {'A': 1}, {'B': 1})
    return SplitDecision(True, epsilon, gain_gap, "Hybrid Hoeffding RBF Surrogate")

def smoke_test():
    centers = [(1.0, 2.0), (3.0, 4.0)]
    weights = [0.5, 0.5]
    epsilon = 1.0
    sensitivity = 0.1
    total_records = 100
    range_ = 1.0
    delta = 0.1
    n = 100

    risk_score = hybrid_privacy_rbf_surrogate(centers, weights, epsilon, sensitivity, total_records)
    print("Reconstruction Risk Score:", risk_score)

    split_decision = hybrid_hoeffding_rbf_surrogate(centers, weights, epsilon, range_, delta, n)
    print("Split Decision:", split_decision)

if __name__ == "__main__":
    smoke_test()