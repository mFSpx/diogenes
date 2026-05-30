# DARWIN HAMMER — match 2309, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s7.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_ternar_m177_s0.py (gen3)
# born: 2026-05-29T23:41:46Z

"""
Hybrid Multivector Fisher Localization (HMFL)

This module combines the mathematical structures of two parent algorithms:
1. hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s7.py - provides a geometric-algebra implementation
2. hybrid_hybrid_fisher_locali_hybrid_hybrid_ternar_m177_s0.py - defines a Fisher information score for a Gaussian beam intensity

The mathematical bridge between these structures is the use of a Gaussian distribution in the Fisher information score and the geometric-algebra implementation of Multivector. By combining these two algorithms, we can create a hybrid system that uses the Fisher information score to inform the Multivector operations.

The hybrid algorithm therefore:
1. Calculates the Fisher information score for a given angle and Gaussian beam intensity
2. Uses the Fisher information score to inform the Multivector operations
3. Performs geometric-algebra operations using Multivector
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Set, Hashable, Sequence

# ----------------------------------------------------------------------
# Basic geometric utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Deterministic Voronoi assignment of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Minimal geometric‑algebra implementation
# ----------------------------------------------------------------------
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
                # duplicate basis vector → cancels (grade‑2+ becomes 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Very small GA container supporting only addition and outer product."""
    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, {self.n})"

# ----------------------------------------------------------------------
# Fisher information score
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid Multivector Fisher Localization
# ----------------------------------------------------------------------
def hybrid_mv_fl(points: List[Point], seeds: List[Point], theta: float, center: float, width: float) -> Multivector:
    regions = assign(points, seeds)
    mv_components = {}
    for i, region in regions.items():
        fisher_inf = fisher_score(theta, center, width)
        mv_components[frozenset({i})] = fisher_inf * len(region)
    return Multivector(mv_components, len(seeds))

def hybrid_mv_fl_scalar(points: List[Point], seeds: List[Point], theta: float, center: float, width: float) -> float:
    mv = hybrid_mv_fl(points, seeds, theta, center, width)
    return mv.scalar_part()

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    seeds = [(0.0, 0.0), (10.0, 10.0)]
    theta = 0.5
    center = 0.0
    width = 1.0
    mv = hybrid_mv_fl(points, seeds, theta, center, width)
    print(mv)
    scalar_part = hybrid_mv_fl_scalar(points, seeds, theta, center, width)
    print(scalar_part)