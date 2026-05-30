# DARWIN HAMMER — match 1224, survivor 3
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# born: 2026-05-29T23:34:33Z

"""
This module implements a hybrid mathematical algorithm that combines the Fisher-information scoring 
from the 'hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0.py' module 
with the Voronoi partitioning and circuit-breaker from the 'hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py' module.

The mathematical bridge between the two structures is based on representing the Fisher-information scoring 
as a method to optimize the Voronoi partitioning process, which is then used to compute the circuit-breaker failure threshold.

The core idea is to use the Fisher-information scoring to optimize the Voronoi partitioning process, 
which is then used to compute the circuit-breaker failure threshold.
"""

import numpy as np
import math
import random
import sys
import pathlib

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: list[tuple[float, float]], 
                            sites: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = ""

    def allow(self) -> bool:
        """True if the circuit is closed."""
        return not self.open

def hybrid_fisher_voronoi(points: list[tuple[float, float]], 
                          sites: list[tuple[float, float]], 
                          theta: float, 
                          center: float, 
                          width: float) -> tuple[dict[int, list[tuple[float, float]]], EndpointCircuitBreaker]:
    """
    Compute Voronoi regions and circuit-breaker failure threshold 
    using Fisher-information scoring as optimization method.
    """
    fisher_info = fisher_score(theta, center, width)
    failure_threshold = int(np.ceil(fisher_info))
    circuit_breaker = EndpointCircuitBreaker(failure_threshold)
    regions = compute_voronoi_regions(points, sites)
    return regions, circuit_breaker

def demonstrate_hybrid_operation():
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    sites = [(0, 0), (2, 2), (4, 4)]
    theta = 1.5
    center = 2.0
    width = 0.5
    regions, circuit_breaker = hybrid_fisher_voronoi(points, sites, theta, center, width)
    print(regions)
    print(circuit_breaker.allow())

if __name__ == "__main__":
    demonstrate_hybrid_operation()