# DARWIN HAMMER — match 784, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m140_s0.py (gen4)
# born: 2026-05-29T23:30:50Z

"""
Hybrid module combining the geometric product and Voronoi partitioning from 
'hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py' and the Fisher 
information and Gaussian beam intensity from 'hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m140_s0.py'.

The mathematical bridge lies in applying the Fisher information to the 
multivectors obtained from the geometric product, and then using the 
Gaussian beam intensity to weight the connectivity between Voronoi partitions.
"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Dict, Any, Tuple

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
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
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        return Multivector({blade: coef for blade, coef in self.components.items() if len(blade) == k}, self.n)

    def scalar_part(self):
        return self.components.get(frozenset(), 0.0)

    def __repr__(self):
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                terms.append(f"{coef}*{blade}")
        return " + ".join(terms)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_fisher_voronoi(points: list[Point], seeds: list[Point], center: float, width: float) -> Dict[int, Multivector]:
    regions = assign(points, seeds)
    fisher_voronoi = {}
    for i, region in regions.items():
        multivector_components = {}
        for point in region:
            theta = math.atan2(point[1], point[0])
            fisher_inf = fisher_score(theta, center, width)
            multivector_components[frozenset({i})] = multivector_components.get(frozenset({i}), 0) + fisher_inf
        fisher_voronoi[i] = Multivector(multivector_components, 2)
    return fisher_voronoi

def estimate_certainty_flag(label: str, confidence_bps: int, authority_class: str, rationale: str, evidence_refs: Tuple[str, ...] = ()) -> Dict[str, Any]:
    return {
        "label": label,
        "confidence_bps": int(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": tuple(str(x) for x in evidence_refs if x is not None),
    }

def hybrid_gaussian_voronoi(points: list[Point], seeds: list[Point], center: float, width: float) -> Dict[int, float]:
    regions = assign(points, seeds)
    gaussian_voronoi = {}
    for i, region in regions.items():
        gaussian_sum = 0
        for point in region:
            theta = math.atan2(point[1], point[0])
            gaussian_sum += gaussian_beam(theta, center, width)
        gaussian_voronoi[i] = gaussian_sum / len(region)
    return gaussian_voronoi

if __name__ == "__main__":
    points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(100)]
    seeds = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(5)]
    center = 0.0
    width = 1.0
    fisher_voronoi = hybrid_fisher_voronoi(points, seeds, center, width)
    gaussian_voronoi = hybrid_gaussian_voronoi(points, seeds, center, width)
    print("Fisher Voronoi:")
    for i, multivector in fisher_voronoi.items():
        print(f"{i}: {multivector}")
    print("Gaussian Voronoi:")
    for i, gaussian in gaussian_voronoi.items():
        print(f"{i}: {gaussian}")