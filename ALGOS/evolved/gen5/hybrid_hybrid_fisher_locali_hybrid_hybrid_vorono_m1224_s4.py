# DARWIN HAMMER — match 1224, survivor 4
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# born: 2026-05-29T23:34:33Z

"""
This module implements a hybrid mathematical algorithm that combines the Fisher-information scoring from the 'hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s0.py' module 
with the Voronoi partitioning and circuit-breaker from the 'hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py' module. 
The mathematical bridge between the two structures is based on representing the Fisher-information scoring as a weighting scheme for the Voronoi regions.

The core idea is to use the Fisher-information scoring to weight the Voronoi regions, which are then used to compute the circuit-breaker failure threshold.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Fisher-information helpers
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Voronoi helpers
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions

# ----------------------------------------------------------------------
# Circuit-breaker
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_voronoi_fisher(points: List[Point], sites: List[Point], 
                           theta: float, center: float, width: float) -> Dict[int, float]:
    """
    Compute Voronoi regions and weight them using Fisher-information scoring.
    """
    regions = compute_voronoi_regions(points, sites)
    weights = {}
    for i, region in regions.items():
        fisher_weight = fisher_score(theta, center, width)
        weights[i] = fisher_weight * len(region)
    return weights

def hybrid_circuit_breaker(weights: Dict[int, float], failure_threshold: int = 3) -> EndpointCircuitBreaker:
    """
    Compute circuit-breaker failure threshold using weighted Voronoi regions.
    """
    total_weight = sum(weights.values())
    threshold = failure_threshold * total_weight
    return EndpointCircuitBreaker(int(threshold))

def hybrid_failure_probability(circuit_breaker: EndpointCircuitBreaker, 
                               weights: Dict[int, float]) -> float:
    """
    Compute failure probability using circuit-breaker and weighted Voronoi regions.
    """
    if circuit_breaker.allow():
        return 0.0
    else:
        total_weight = sum(weights.values())
        return circuit_breaker.failures / total_weight

if __name__ == "__main__":
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]
    sites = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]
    theta, center, width = 5.0, 5.0, 1.0
    weights = hybrid_voronoi_fisher(points, sites, theta, center, width)
    circuit_breaker = hybrid_circuit_breaker(weights)
    failure_prob = hybrid_failure_probability(circuit_breaker, weights)
    print(failure_prob)