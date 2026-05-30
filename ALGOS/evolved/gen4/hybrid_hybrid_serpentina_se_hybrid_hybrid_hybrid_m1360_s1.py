# DARWIN HAMMER — match 1360, survivor 1
# gen: 4
# parent_a: hybrid_serpentina_self_righ_xgboost_objective_m78_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_pherom_m30_s0.py (gen3)
# born: 2026-05-29T23:35:47Z

import math
import random
import sys
import pathlib
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, FrozenSet

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

def binary_logistic_grad_hess(y_true: np.ndarray,
                              margin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)),
               key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def shannon_entropy(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs if p > 0)

def ollivier_ricci_curvature(seeds: List[Point],
                             probs: List[float]) -> float:
    if len(seeds) < 2:
        return 0.0
    eps = 1e-12
    curvatures = []
    n = len(seeds)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(seeds[i], seeds[j])
            prob_diff = abs(probs[i] - probs[j])
            curv = 1.0 - prob_diff / (d + eps)
            curvatures.append(curv)
    return sum(curvatures) / len(curvatures)

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            lst.pop(i)
            lst.pop(i)
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int],
                     blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    sorted_indices, sign = _blade_sign(combined)
    return frozenset(sorted_indices), sign

class Multivector:
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = self.components.copy()
        for blade, val in other.components.items():
            result[blade] = result.get(blade, 0.0) + val
        return Multivector(result, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
        return Multivector(result, self.n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

def morphology_to_multivector(m: Morphology) -> Multivector:
    comps = {
        frozenset({1}): m.length,
        frozenset({2}): m.width,
        frozenset({3}): m.height,
        frozenset({4}): m.mass,
    }
    return Multivector(comps, n=4)

def pheromone_region_entropy(points: List[Point],
                             seeds: List[Point]) -> Tuple[float, List[float]]:
    regions = assign(points, seeds)
    total = sum(len(v) for v in regions.values())
    if total == 0:
        probs = [1.0 / len(seeds)] * len(seeds)
        return 0.0, probs
    probs = [len(regions[i]) / total for i in range(len(seeds))]
    entropy = shannon_entropy(probs)
    return entropy, probs

def hybrid_recovery_score(m: Morphology,
                          points: List[Point],
                          seeds: List[Point]) -> float:
    r = recovery_priority(m)
    H, probs = pheromone_region_entropy(points, seeds)
    H_max = math.log(len(seeds)) if len(seeds) > 1 else 0.0
    kappa = ollivier_ricci_curvature(seeds, probs)
    p_hat = r * (1 - H / (H_max + 1e-12)) * (1 + kappa) / 2
    return max(0.0, min(1.0, p_hat))