# DARWIN HAMMER — match 1117, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s4.py (gen3)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py (gen2)
# born: 2026-05-29T23:32:56Z

import math
import numpy as np
from typing import List, Tuple, Dict

def _gamma(z: float) -> float:
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857
    ])
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)


def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: Tuple[float, ...], seeds: List[Tuple[float, ...]]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[Tuple[float, ...]], points: List[Tuple[float, ...]]) -> Dict[int, List[Tuple[float, ...]]]:
    regions: Dict[int, List[Tuple[float, ...]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def hybrid_voronoi_fractional(points: List[Tuple[float, ...]], seeds: List[Tuple[float, ...]], alpha: float, times: np.ndarray) -> Dict[int, Dict[Tuple[float, ...], float]]:
    regions = voronoi_partition(seeds, points)
    kernel = caputo_kernel(alpha, times)
    weighted_regions = {}
    for seed_idx, region in regions.items():
        weighted_region = {}
        for point in region:
            weight = np.dot(kernel, [euclidean_distance(point, seeds[seed_idx])] * len(times))
            weighted_region[point] = weight
        weighted_regions[seed_idx] = weighted_region
    return weighted_regions


def hybrid_voronoi_partition(points: List[Tuple[float, ...]], seeds: List[Tuple[float, ...]]) -> Dict[int, List[Tuple[float, ...]]]:
    return voronoi_partition(seeds, points)


def hybrid_caputo_weights(alpha: float, times: np.ndarray) -> np.ndarray:
    return caputo_kernel(alpha, times)


def improved_hybrid_voronoi_fractional(points: List[Tuple[float, ...]], seeds: List[Tuple[float, ...]], alpha: float, times: np.ndarray) -> Dict[int, Dict[Tuple[float, ...], np.ndarray]]:
    regions = voronoi_partition(seeds, points)
    kernel = caputo_kernel(alpha, times)
    weighted_regions = {}
    for seed_idx, region in regions.items():
        weighted_region = {}
        for point in region:
            distances = np.array([euclidean_distance(point, seeds[seed_idx])] * len(times))
            weight = np.dot(kernel, distances)
            weighted_region[point] = np.array([weight * t for t in times])
        weighted_regions[seed_idx] = weighted_region
    return weighted_regions


if __name__ == "__main__":
    points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    seeds = [(0, 0), (2, 2)]
    alpha = 0.5
    times = np.arange(1, 10)
    weighted_regions = improved_hybrid_voronoi_fractional(points, seeds, alpha, times)
    for seed_idx, region in weighted_regions.items():
        print(f"Seed {seed_idx}:")
        for point, weight in region.items():
            print(f"  {point}: {weight}")
    print(hybrid_voronoi_partition(points, seeds))
    print(hybrid_caputo_weights(alpha, times))