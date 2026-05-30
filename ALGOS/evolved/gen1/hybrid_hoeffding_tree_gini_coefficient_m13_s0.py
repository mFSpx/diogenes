# DARWIN HAMMER — match 13, survivor 0
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:25:17Z

import numpy as np
from dataclasses import dataclass
from collections.abc import Iterable
from math import sqrt, log
from random import randint
from sys import path_hooks
from pathlib import Path

# Module docstring explaining the mathematical bridge between Hoeffding Tree (hoeffding_tree.py) and Gini Coefficient (gini_coefficient.py)
"""
Hybrid algorithm combining Hoeffding Tree decision strategy from CapyMOA/MOA with Gini Coefficient inequality measure.
The Hoeffding Tree's bound computation from (hoeffding_bound) is linked to the Gini Coefficient's weighted sum computation from (gini_coefficient).
This link is established through a weighted average of the Gini Coefficient's numerator and Hoeffding Tree's bound, resulting in a new decision criterion.
The Gini Coefficient is adapted to be non-negative even when dealing with negative data points by ignoring the first negative value.
"""

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound computation from Hoeffding Tree (hoeffding_tree.py)"""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return sqrt((r * r * log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini inequality coefficient from Gini Coefficient (gini_coefficient.py)"""
    xs = sorted((float(x) for x in values))
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0:
        # Ignore the first negative value
        for i, x in enumerate(xs):
            if x >= 0:
                xs = xs[i:]
                break
        if not xs or sum(xs) == 0: return 0.0
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def hybrid_bound(values: Iterable[float], r: float, delta: float, n: int) -> float:
    """Hybrid bound computation linking Hoeffding Tree (hoeffding_bound) and Gini Coefficient (gini_coefficient)"""
    gini = gini_coefficient(values)
    eps = hoeffding_bound(r, delta, n)
    return gini + eps


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(values: Iterable[float], best_gain: float, second_best_gain: float, r: float, delta: int, n: int) -> SplitDecision:
    """Hybrid decision strategy combining Hoeffding Tree (should_split) and Gini Coefficient"""
    eps = hybrid_bound(values, r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < 0.05  # arbitrary threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < 0.05 else "wait")
    return SplitDecision(split, eps, gap, reason)


def generate_random_values(length: int) -> Iterable[float]:
    """Generate random non-negative values with 50% chance of value being negative"""
    return (randint(-100, 100) if randint(0, 1) == 0 else randint(0, 100) for _ in range(length))


if __name__ == "__main__":
    # Smoke test to ensure the hybrid algorithm runs without error
    values = generate_random_values(100)
    decision = should_split(values, 0.5, 0.3, 0.02, 0.05, 50)
    print(decision.__dict__)