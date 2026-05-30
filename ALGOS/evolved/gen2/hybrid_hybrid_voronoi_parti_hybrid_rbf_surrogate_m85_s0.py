# DARWIN HAMMER — match 85, survivor 0
# gen: 2
# parent_a: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s1.py (gen1)
# parent_b: hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py (gen1)
# born: 2026-05-29T23:26:42Z

"""
Module hybrid_voronoi_rbf_surrogate: A hybrid algorithm combining the Voronoi partition
from hybrid_voronoi_partition_poikilotherm_schoolf_m49_s1.py and the radial-basis surrogate
model from hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py. The mathematical bridge between
the two structures lies in the use of Voronoi cell centers as surrogate model centers,
enabling the use of Gaussian radial basis functions to model the signal scores and noise
scores from the Voronoi partition, effectively creating a probabilistic surrogate model
for decision-making.
"""

import math
import numpy as np
import random
import sys

from typing import Tuple, List, Dict
from dataclasses import dataclass

Vector = List[float]

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def fit(points: List[Vector], values: List[float], epsilon: float = 1.0, ridge: float = 1e-9) -> Dict[Tuple[float, float], float]:
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non-empty and same length")
    k = [[gaussian(euclidean(a, b), epsilon) + (ridge if i == j else 0.0) for j, b in enumerate(centers)] for i, a in enumerate(centers)]
    surrogate = solve_linear(k, y)
    return {c: s for c, s in zip(centers, surrogate)}

def assign_hybrid_region(points: List[Tuple[float, float]], seeds1: List[Tuple[float, float]], seeds2: List[Tuple[float, float]]) -> Dict[int, Dict[Tuple[float, float], float]]:
    assigned_region = assign(points, seeds1)
    thermal_rates = [normalized_activity(tuple(map(float, k))[0]) for k in assigned_region.keys()]
    surrogate_model = fit([(k[0], k[1]) for k in assigned_region.keys()], thermal_rates)
    hybrid_regions = {}
    for region, points in assigned_region.items():
        region_surrogates = {}
        for p in points:
            region_surrogates[p] = surrogate_model.get(p, 0.0)
        hybrid_regions[region] = region_surrogates
    return hybrid_regions

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

if __name__ == "__main__":
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seeds1 = [(0.0, 0.0), (4.0, 4.0)]
    seeds2 = [(5.0, 5.0), (6.0, 6.0)]
    hybrid_regions = assign_hybrid_region(points, seeds1, seeds2)
    print(hybrid_regions)