# DARWIN HAMMER — match 4394, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2101_s1.py (gen5)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s5.py (gen2)
# born: 2026-05-29T23:55:17Z

"""
Module for hybrid algorithm fusing concepts from 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2101_s1 and 
hybrid_fisher_localization_hybrid_ternary_route_m40_s5.

The mathematical bridge between the two parents lies in the use of structural 
similarity (SSIM) and Gaussian beam intensity. The hybrid algorithm integrates 
these concepts by applying weighted SSIM to Multivectors and utilizing the 
Fisher score as a metric for evaluating the similarity between Multivectors.

This module provides functions for computing weighted SSIM, Fisher score, and 
performing operations on Multivectors.
"""

import numpy as np
import math
import random
from pathlib import Path
from typing import Tuple, List, Dict, Sequence

Point = Tuple[float, float]
Node = object
Graph = Dict[Node, set[Node]]
FeatureVec = List[float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                n -= 1
                break
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    def __init__(self, components: List[float], n: int):
        self.components = components
        self.n = n

    def __add__(self, other):
        if self.n != other.n:
            raise ValueError("Multivectors must have the same dimension")
        return Multivector([a + b for a, b in zip(self.components, other.components)], self.n)

    def __mul__(self, other):
        if self.n != other.n:
            raise ValueError("Multivectors must have the same dimension")
        return Multivector([a * b for a, b in zip(self.components, other.components)], self.n)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def weighted_ssim(
    x: Sequence[float],
    y: Sequence[float],
    theta: float,
    center: float,
    width: float,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    # Compute per-sample weights
    weights = [gaussian_beam(i, center, width) for i in range(len(x))]

    # Normalize weights
    sum_weights = sum(weights)
    weights = [w / sum_weights for w in weights]

    # Compute weighted SSIM
    mean_x = sum(x[i] * weights[i] for i in range(len(x)))
    mean_y = sum(y[i] * weights[i] for i in range(len(y)))

    variance_x = sum((x[i] - mean_x) ** 2 * weights[i] for i in range(len(x)))
    variance_y = sum((y[i] - mean_y) ** 2 * weights[i] for i in range(len(y)))

    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) * weights[i] for i in range(len(x)))

    c1 = (k1 * dynamic_range) ** 2 if dynamic_range else (k1 * max(x + y)) ** 2
    c2 = (k2 * dynamic_range) ** 2 if dynamic_range else (k2 * max(x + y)) ** 2

    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x ** 2 + mean_y ** 2 + c1) * (variance_x + variance_y + c2))

    return ssim

def multivector_ssim(
    mv1: Multivector,
    mv2: Multivector,
    theta: float,
    center: float,
    width: float,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    return weighted_ssim(mv1.components, mv2.components, theta, center, width, dynamic_range, k1, k2)

def multivector_fisher_score(
    mv1: Multivector,
    mv2: Multivector,
    theta: float,
    center: float,
    width: float,
    eps: float = 1e-12,
) -> float:
    return fisher_score(theta, center, width, eps)

if __name__ == "__main__":
    mv1 = Multivector([1.0, 2.0, 3.0], 3)
    mv2 = Multivector([4.0, 5.0, 6.0], 3)

    print(multivector_ssim(mv1, mv2, 0.5, 0.0, 1.0))
    print(multivector_fisher_score(mv1, mv2, 0.5, 0.0, 1.0))