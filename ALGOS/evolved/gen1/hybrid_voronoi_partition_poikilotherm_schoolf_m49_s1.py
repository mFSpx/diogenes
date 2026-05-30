# DARWIN HAMMER — match 49, survivor 1
# gen: 1
# parent_a: voronoi_partition.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:25:30Z

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

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

def assign_hybrid_region(points: List[Tuple[float, float]], seeds1: List[Tuple[float, float]], seeds2: List[Tuple[float, float]]) -> Dict[int, Dict[int, List[Tuple[float, float]]]]:
    assigned_region = assign(points, seeds1)
    thermal_rates = [normalized_activity(p[0]) for p in points]

    hybrid_regions = {i: {j: [] for j in range(len(seeds2))} for i in range(len(seeds1))}
    for i, p in enumerate(points):
        voronoi_seed_idx = nearest(p, seeds1)
        thermal_region_idx = 0 if thermal_rates[i] < 0.2 else 1 if thermal_rates[i] < 0.5 else 2
        hybrid_regions[voronoi_seed_idx][thermal_region_idx].append(p)

    return assigned_region, hybrid_regions

def main():
    np.random.seed(0)
    points = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(100)]
    seeds1 = [(50, 50)]
    seeds2 = [(10, 10), (90, 90)]

    assigned_region, hybrid_region = assign_hybrid_region(points, seeds1, seeds2)

    print("Voronoi Region", assigned_region)
    print("Hybrid Region")
    for voronoi_seed_idx, thermal_regions in hybrid_region.items():
        print(f"Voronoi Seed {voronoi_seed_idx}:")
        for thermal_region_idx, region in thermal_regions.items():
            print(f"  Thermal Region {thermal_region_idx}: {region}")

if __name__ == "__main__":
    main()