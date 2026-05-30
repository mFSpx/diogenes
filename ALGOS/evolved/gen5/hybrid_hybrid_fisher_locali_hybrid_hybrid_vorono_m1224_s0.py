# DARWIN HAMMER — match 1224, survivor 0
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# born: 2026-05-29T23:34:33Z

"""
This module implements a novel hybrid mathematical algorithm that combines the Fisher-information scoring 
from the 'hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0' module with the Voronoi partitioning 
and multivector operations from the 'hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5' module. 
The mathematical bridge between the two structures is based on representing the Voronoi regions as 
a set of Gaussian beams, where each beam is optimized using the Fisher-information scoring. 
The Fisher-information scoring is used to optimize the Gaussian beams, which are then used to compute 
the Voronoi regions and perform multivector operations.

The core idea is to use the Fisher-information scoring to optimize the Gaussian beams, which are then 
used to compute the Voronoi regions and perform multivector operations.
"""

import numpy as np
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = tuple[float, float]
Blade = frozenset[int]          # basis blade represented by a set of indices
Multivector = dict[Blade, float]  # mapping blade → coefficient

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: list[Point], sites: list[Point]) -> dict[int, list[Point]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: dict[int, list[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

def optimize_gaussian_beams(points: list[Point], sites: list[Point]) -> dict[int, float]:
    """
    Optimize the Gaussian beams using the Fisher-information scoring.
    Returns a dict ``site_index → optimized_width``.
    """
    optimized_widths: dict[int, float] = {}
    for i, site in enumerate(sites):
        points_in_region = [pt for pt in points if euclidean_distance(pt, site) < euclidean_distance(pt, sites[(i+1)%len(sites)])]
        if points_in_region:
            center = np.mean([pt[0] for pt in points_in_region])
            width = np.std([pt[0] for pt in points_in_region])
            optimized_widths[i] = width
        else:
            optimized_widths[i] = 1.0
    return optimized_widths

def compute_multivector(points: list[Point], sites: list[Point], optimized_widths: dict[int, float]) -> Multivector:
    """
    Compute the multivector using the optimized Gaussian beams.
    Returns a dict ``blade → coefficient``.
    """
    multivector: Multivector = {}
    for i, site in enumerate(sites):
        points_in_region = [pt for pt in points if euclidean_distance(pt, site) < euclidean_distance(pt, sites[(i+1)%len(sites)])]
        if points_in_region:
            blade = frozenset([i])
            coefficient = np.sum([gaussian_beam(pt[0], site[0], optimized_widths[i]) for pt in points_in_region])
            multivector[blade] = coefficient
    return multivector

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    sites = [(random.random(), random.random()) for _ in range(5)]
    regions = compute_voronoi_regions(points, sites)
    optimized_widths = optimize_gaussian_beams(points, sites)
    multivector = compute_multivector(points, sites, optimized_widths)
    print("Voronoi Regions:", regions)
    print("Optimized Widths:", optimized_widths)
    print("Multivector:", multivector)