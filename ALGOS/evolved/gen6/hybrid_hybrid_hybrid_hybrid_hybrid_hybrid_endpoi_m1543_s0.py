# DARWIN HAMMER — match 1543, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m614_s1.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s5.py (gen2)
# born: 2026-05-29T23:37:13Z

"""
Hybrid module combining the geometric product, Voronoi partitioning, 
Ollivier-Ricci curvature calculation from 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m614_s1.py' 
and the path signature calculation, lead-lag transformation from 'hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s5.py'. 
The mathematical bridge lies in applying the lead-lag transformation to the Voronoi partitions 
and then using the Ollivier-Ricci curvature calculation to quantify the connectivity 
between the resulting geometric objects, while also integrating the path signature calculation 
to analyze the topology of the Voronoi diagram.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter

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

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def voronoi_signature(points: list[Point], seeds: list[Point]) -> dict[int, np.ndarray]:
    regions = assign(points, seeds)
    signatures = {}
    for i, region in regions.items():
        region_array = np.array(region)
        signatures[i] = signature_level1(lead_lag_transform(region_array))
    return signatures

def ollivier_ricci_curvature(points: list[Point], seeds: list[Point]) -> float:
    regions = assign(points, seeds)
    total_curvature = 0
    for region in regions.values():
        region_array = np.array(region)
        curvature = np.sum(np.abs(lead_lag_transform(region_array)))
        total_curvature += curvature
    return total_curvature / len(regions)

def hybrid_operation(points: list[Point], seeds: list[Point]) -> tuple[dict[int, np.ndarray], float]:
    signatures = voronoi_signature(points, seeds)
    curvature = ollivier_ricci_curvature(points, seeds)
    return signatures, curvature

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    signatures, curvature = hybrid_operation(points, seeds)
    print(signatures)
    print(curvature)