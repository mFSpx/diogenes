# DARWIN HAMMER — match 13, survivor 3
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:25:17Z

#!/usr/bin/env python3
"""Hybrid Hoeffding-Gini algorithm, combining the Hoeffding bound helpers for stream splits from hoeffding_tree.py 
and the Gini inequality coefficient from gini_coefficient.py. The mathematical bridge is formed by using the Gini 
coefficient to evaluate the goodness of split, and the Hoeffding bound to determine when to split based on the Gini 
gain. This creates a self-adjusting decision tree that balances exploration and exploitation."""

import math
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import random
import sys
import pathlib

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gini: float, second_best_gini: float, r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gini - second_best_gini
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def evaluate_split(gini_values: Iterable[float], r: float, delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    gini_coeff = gini_coefficient(gini_values)
    best_gini = gini_coeff
    second_best_gini = 0.0
    for gini in gini_values:
        if gini > best_gini:
            second_best_gini = best_gini
            best_gini = gini
        elif gini > second_best_gini:
            second_best_gini = gini
    return should_split(best_gini, second_best_gini, r, delta, n, tie_threshold)

def generate_random_gini_values(num_values: int, max_value: float = 1.0) -> Iterable[float]:
    return [random.uniform(0, max_value) for _ in range(num_values)]

if __name__ == "__main__":
    num_values = 10
    max_value = 1.0
    r = 1.0
    delta = 0.1
    n = 100
    tie_threshold = 0.05
    gini_values = generate_random_gini_values(num_values, max_value)
    split_decision = evaluate_split(gini_values, r, delta, n, tie_threshold)
    print(split_decision)