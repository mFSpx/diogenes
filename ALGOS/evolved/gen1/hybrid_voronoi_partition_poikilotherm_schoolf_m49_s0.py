# DARWIN HAMMER — match 49, survivor 0
# gen: 1
# parent_a: voronoi_partition.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:25:30Z

import math
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass

def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
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
    """Map an observed operating temperature to a 0..1 activity gate."""
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))


def distance_seeds(seeds1: list[tuple[float, float]], seeds2: list[tuple[float, float]]) -> float:
    return distance(seeds1[0], seeds2[0])


def assign_thermal_region(points: list[tuple[float, float]], seeds1: list[tuple[float, float]], seeds2: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    assigned_region = assign(points, seeds1)
    thermal_rate = [normalized_activity(p[0], 15.0, 30.0) for p in points]
    thermal_regions = {i: [] for i in range(len(seeds2))}
    for i, p in enumerate(points):
        thermal_regions[0].append(p) if thermal_rate[i] < 0.2 else \
            thermal_regions[1].append(p) if thermal_rate[i] < 0.5 else \
            thermal_regions[2].append(p)
    return assigned_region, thermal_regions


def assign_hybrid_region(points: list[tuple[float, float]], seeds1: list[tuple[float, float]], seeds2: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    # Calculate Voronoi regions
    assigned_region = assign(points, seeds1)

    # Create temperature-dependent thermal regions
    thermal_rate = [normalized_activity(p[0]) for p in points]
    thermal_regions = {i: [] for i in range(len(seeds2))}
    for i, p in enumerate(points):
        thermal_regions[0].append(p) if thermal_rate[i] < 0.2 else thermal_regions[1].append(p)

    # Fuse Voronoi and thermal regions
    fused_regions = {i: [] for i in range(len(seeds2))}
    for i, voronoi_seed in enumerate(seeds1):
        for j, thermal_seed in enumerate(seeds2):
            distance_voronoi = distance_seeds([voronoi_seed, thermal_seed])
            fused_regions[j].append(points[np.argmin([distance(p, voronoi_seed) + 0.1 * abs(thermal_rate[i] - thermal_rate[k]) for k, p in enumerate(points)])])

    return assigned_region, thermal_regions, fused_regions


def main():
    # Generate random test data
    np.random.seed(0)
    points = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(100)]
    seeds1 = [(50, 50)]
    seeds2 = [(10, 10), (90, 90)]

    # Assign points to Voronoi regions
    assigned_region = assign(points, seeds1)

    # Assign points to thermal regions
    thermal_region = {i: [] for i in range(len(seeds2))}
    for p in points:
        thermal_region[0].append(p) if normalized_activity(p[0]) < 0.5 else thermal_region[1].append(p)

    # Assign points to hybrid regions
    hybrid_region = assign_hybrid_region(points, seeds1, seeds2)

    print("Voronoi Region", assigned_region)
    print("Thermal Region", thermal_region)
    print("Hybrid Region", hybrid_region)


if __name__ == "__main__":
    main()