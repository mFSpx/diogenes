# DARWIN HAMMER — match 892, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_rbf_surrogate_m85_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py (gen4)
# born: 2026-05-29T23:31:25Z

"""
Module hybrid_voronoi_rbf_decision_fusion: A hybrid algorithm combining the Voronoi partition
and radial-basis surrogate model from hybrid_hybrid_voronoi_parti_hybrid_rbf_surrogate_m85_s0.py
with the decision-making and Shapley value computation from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py.
The mathematical bridge between the two structures lies in the use of Voronoi cell centers as 
surrogate model centers for decision-making, and the application of Shapley value computation 
to evaluate feature contributions in the Voronoi-partitioned space.

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

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    if subset_size < 0 or subset_size > feature_count:
        raise ValueError("Subset size must be between 0 and feature count.")
    return math.comb(feature_count, subset_size) * (subset_size - 1) / feature_count

def hybrid_voronoi_rbf_decision(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], 
                                feature_values: List[float], feature_count: int) -> Dict[int, Dict[str, float]]:
    regions = assign(points, seeds)
    result = {}
    for region_idx, region_points in regions.items():
        region_center = seeds[region_idx]
        rbf_values = [np.exp(-distance(point, region_center)**2) for point in region_points]
        shapley_values = [shapley_kernel_weight(i, feature_count) * feature_values[i] for i in range(feature_count)]
        result[region_idx] = {
            'rbf_values': rbf_values,
            'shapley_values': shapley_values,
            'sphericity': sphericity_index(*[max(abs(x - region_center[0]), 1e-6), max(abs(x - region_center[1]), 1e-6), 1.0])
        }
    return result

def evaluate_hybrid_model(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]], 
                          feature_values: List[float], feature_count: int) -> float:
    result = hybrid_voronoi_rbf_decision(points, seeds, feature_values, feature_count)
    total_shapley_value = 0.0
    for region_idx, region_info in result.items():
        total_shapley_value += sum(region_info['shapley_values'])
    return total_shapley_value / feature_count

if __name__ == "__main__":
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]
    seeds = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]
    feature_values = [random.uniform(0, 1) for _ in range(3)]
    feature_count = len(feature_values)
    print(evaluate_hybrid_model(points, seeds, feature_values, feature_count))