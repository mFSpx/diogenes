# DARWIN HAMMER — match 5839, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_workshare_all_m1367_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1209_s1.py (gen6)
# born: 2026-05-30T00:04:53Z

"""
Hybrid module combining the Schoolfield model and geometric product from 'hybrid_hybrid_hybrid_bandit_hybrid_workshare_all_m1367_s2.py' 
and 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1209_s1.py'. 
The mathematical bridge lies in applying the Schoolfield model to quantify the developmental rate 
of points in a Voronoi partitioning and using the geometric product to calculate the multivector 
representation of the points.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

@dataclass(frozen=True)
class Point:
    x: float
    y: float

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator * (low + high)

def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

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

def calculate_developmental_rate(points: List[Point], seeds: List[Point], temp_c: float) -> Dict[int, float]:
    temp_k = c_to_k(temp_c)
    schoolfield_params = SchoolfieldParams()
    regions = assign(points, seeds)
    rates = {}
    for i, region in regions.items():
        rate = 0.0
        for point in region:
            rate += developmental_rate(temp_k, schoolfield_params)
        rates[i] = rate / len(region)
    return rates

def calculate_multivector(points: List[Point], seeds: List[Point]) -> Multivector:
    regions = assign(points, seeds)
    components = {}
    for i, region in regions.items():
        blade = frozenset([point.x for point in region] + [point.y for point in region])
        components[blade] = len(region)
    return Multivector(components, len(points))

def hybrid_operation(points: List[Point], seeds: List[Point], temp_c: float) -> Dict[int, float]:
    temp_k = c_to_k(temp_c)
    schoolfield_params = SchoolfieldParams()
    regions = assign(points, seeds)
    rates = {}
    for i, region in regions.items():
        rate = 0.0
        for point in region:
            rate += developmental_rate(temp_k, schoolfield_params)
        rates[i] = rate / len(region)
    multivector = calculate_multivector(points, seeds)
    return rates, multivector

if __name__ == "__main__":
    points = [Point(1.0, 2.0), Point(3.0, 4.0), Point(5.0, 6.0)]
    seeds = [Point(0.0, 0.0), Point(10.0, 10.0)]
    temp_c = 20.0
    rates, multivector = hybrid_operation(points, seeds, temp_c)
    print(rates)
    print(multivector.components)