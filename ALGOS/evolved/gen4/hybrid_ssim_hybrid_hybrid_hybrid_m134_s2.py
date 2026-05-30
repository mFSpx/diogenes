# DARWIN HAMMER — match 134, survivor 2
# gen: 4
# parent_a: ssim.py (gen0)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py (gen3)
# born: 2026-05-29T23:27:03Z

"""
This module fuses the Structural Similarity Index (SSIM) from ssim.py and the 
Hybrid Decision Hygiene and Geometric Algebra from hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py.
The mathematical bridge is established by representing the SSIM as a multivector 
in a Clifford algebra, where each component of the SSIM (mean, variance, covariance) 
is associated with a basis vector. The geometric product and inner product of these 
multivectors can be used to analyze and compare the structural similarity and decision 
hygiene scores in a more nuanced and expressive way.

Parents:
- Structural Similarity Index (SSIM): ssim.py
- Hybrid Decision Hygiene and Geometric Algebra: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import math
import random
import sys
import pathlib

class Multivector:
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items()})

def ssim_to_multivector(x: List[float], y: List[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> Multivector:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    ssim_val = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return Multivector({frozenset(): ssim_val, frozenset({0}): mx, frozenset({1}): my, frozenset({0, 1}): cov}, 2)

def geometric_product(mv1: Multivector, mv2: Multivector) -> Multivector:
    result = Multivector({}, mv1.n)
    for blade1, coef1 in mv1.components.items():
        for blade2, coef2 in mv2.components.items():
            blade = blade1 ^ blade2
            coef = coef1 * coef2
            result.components[blade] = result.components.get(blade, 0.0) + coef
    return result

def inner_product(mv1: Multivector, mv2: Multivector) -> float:
    return sum(mv1.components.get(blade, 0.0) * mv2.components.get(blade, 0.0) for blade in set(mv1.components) | set(mv2.components))

def hybrid_ssim(x: List[float], y: List[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> Tuple[float, Multivector]:
    mv1 = ssim_to_multivector(x, y, dynamic_range, k1, k2)
    mv2 = Multivector({frozenset(): 1.0}, 2)
    gp = geometric_product(mv1, mv2)
    ip = inner_product(mv1, mv2)
    return ip, gp

if __name__ == "__main__":
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 3.0, 4.0, 5.0, 6.0]
    ip, gp = hybrid_ssim(x, y)
    print(f"Inner product: {ip}")
    print(f"Geometric product: {gp}")