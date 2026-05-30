# DARWIN HAMMER — match 13, survivor 1
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:25:17Z

#!/usr/bin/env python3
"""Hybrid Hoeffding Bound and Gini Coefficient algorithm, combining the strengths of both CapyMOA/MOA-style Hoeffding bound helpers and Gini inequality coefficient.

This module mathematically interfaces Hoeffding's inequality with the Gini coefficient by leveraging the concept of variance, which is crucial in both algorithms. Hoeffding's inequality measures the probability of a binomial distribution deviating from its mean, while the Gini coefficient measures the variance of a discrete distribution. By combining these two concepts, we create a hybrid algorithm that can adaptively adjust its decision-making criteria based on the underlying data distribution.

We achieve this by modifying the Hoeffding bound calculation to incorporate the Gini coefficient as a regularization term. This allows the algorithm to balance the trade-off between exploration and exploitation, while also accounting for the underlying data distribution.

"""
import math
from dataclasses import dataclass
import numpy as np
import random
import sys
import pathlib

def hoeffding_bound_with_gini(r: float, delta: float, n: int, gini_coeff: float) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    
    # Regularize the Hoeffding bound with the Gini coefficient
    regularization_term = gini_coeff * math.pi / 6
    return math.sqrt((r * r * math.log(1.0 / delta) + regularization_term) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split_with_gini(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, tie_threshold: float = 0.05, gini_coeff: float = 0.5) -> SplitDecision:
    eps = hoeffding_bound_with_gini(r, delta, n, gini_coeff)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


def gini_importance_split(values: list[float], threshold: float) -> float:
    """Calculate the Gini importance of a split based on the weighted Gini coefficients."""
    left_values = [x for x in values if x <= threshold]
    right_values = [x for x in values if x > threshold]
    
    left_gini = gini_coefficient(left_values)
    right_gini = gini_coefficient(right_values)
    
    return (len(left_values) / len(values)) * left_gini + (len(right_values) / len(values)) * right_gini


def hybrid_split_decider(values: list[float], best_threshold: float, second_best_threshold: float, r: float, delta: float, n: int, tie_threshold: float = 0.05, gini_coeff: float = 0.5) -> SplitDecision:
    best_gain = gini_importance_split(values, best_threshold)
    second_best_gain = gini_importance_split(values, second_best_threshold)
    
    eps = hoeffding_bound_with_gini(r, delta, n, gini_coeff)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    random.seed(0)
    values = np.random.rand(100)
    best_threshold = np.median(values)
    second_best_threshold = np.percentile(values, 75)
    
    r = 0.1
    delta = 0.01
    n = 100
    tie_threshold = 0.05
    gini_coeff = 0.5
    
    decision = hybrid_split_decider(values.tolist(), best_threshold, second_best_threshold, r, delta, n, tie_threshold, gini_coeff)
    print(decision)