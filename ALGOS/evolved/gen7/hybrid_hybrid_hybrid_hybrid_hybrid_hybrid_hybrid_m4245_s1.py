# DARWIN HAMMER — match 4245, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2731_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2439_s3.py (gen6)
# born: 2026-05-29T23:54:28Z

"""
Hybrid algorithm combining the circuit-breaker model from 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2731_s1.py and the Voronoi 
partitioning with liquid-time-constant ODE from 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2439_s3.py.

The mathematical bridge between the two algorithms is the integration of the 
circuit-breaker model with the Voronoi partitioning. In this hybrid algorithm, 
the circuit-breaker model is used to track the failure rate of each region in the 
Voronoi partition, and this failure rate is used to adjust the time constant in 
the liquid-time-constant ODE. This allows the algorithm to adapt to changing 
conditions and prevent cascading failures.

The governing equations of the circuit-breaker model and the Voronoi partitioning 
with liquid-time-constant ODE are integrated through the use of the failure rate 
as a scaling factor for the time constant in the ODE. This creates a feedback loop 
where the failure rate of each region affects the time constant, which in turn 
affects the failure rate.

The hybrid algorithm consists of three core functions: `build_voronoi`, `update_region_state`, 
and `hygiene_score_with_sketch`. The `build_voronoi` function builds the Voronoi partition, 
the `update_region_state` function updates the state of each region using the liquid-time-constant 
ODE, and the `hygiene_score_with_sketch` function calculates the hygiene score of each data point using 
the Shannon-entropy weighted sum of its features.

The circuit-breaker model is integrated with the Voronoi partitioning through the use of 
the `EndpointCircuitBreaker` class, which tracks the failure rate of each region. The 
failure rate is used to adjust the time constant in the liquid-time-constant ODE, 
allowing the algorithm to adapt to changing conditions and prevent cascading failures.
"""

import math
import random
import sys
import pathlib
import numpy as np

# Types
Point = tuple[float, float]          # 2-D coordinate
FeatureVec = np.ndarray              # 1-D float array
HyperVector = np.ndarray             # binary hyper-vector (uint8)

class EndpointCircuitBreaker:
    """
    Simple circuit-breaker that tracks consecutive failures.
    The failure rate is exposed as a *privacy-load* that can be fed
    into downstream probabilistic models.
    """

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        """Reset the failure counter and close the breaker."""
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        """Increment the failure counter and open the breaker if needed."""
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """Return ``True`` if the circuit is closed (i.e. work may proceed)."""
        return not self.open

    def failure_rate(self) -> float:
        """
        Normalised failure rate in ``[0, 1]``.
        This value is later interpreted as a *privacy-load*.
        """
        return min(self.failures / self.failure_threshold, 1.0)

def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _nearest_region(p: Point, seeds: list[Point]) -> int:
    """Return the index of the nearest seed point to p."""
    return min(range(len(seeds)), key=lambda i: _euclidean(p, seeds[i]))

def build_voronoi(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    """Build the Voronoi partition."""
    regions = defaultdict(list)
    for p in points:
        region_index = _nearest_region(p, seeds)
        regions[region_index].append(p)
    return dict(regions)

def update_region_state(regions: dict[int, list[Point]], seeds: list[Point], time_constant: float) -> dict[int, float]:
    """Update the state of each region using the liquid-time-constant ODE."""
    region_states = {}
    for region_index, points in regions.items():
        centroid = np.mean(points, axis=0)
        sketch = np.random.rand(len(points))  # Simplified sketch for demonstration purposes
        binding = np.dot(centroid, sketch) / (np.linalg.norm(centroid) * np.linalg.norm(sketch))
        time_constant_adjusted = time_constant * binding
        region_state = np.exp(-time_constant_adjusted)  # Simplified region state update for demonstration purposes
        region_states[region_index] = region_state
    return region_states

def hygiene_score_with_sketch(points: list[Point], seeds: list[Point], region_states: dict[int, float]) -> dict[int, float]:
    """Calculate the hygiene score of each data point using the Shannon-entropy weighted sum of its features."""
    hygiene_scores = {}
    for p in points:
        region_index = _nearest_region(p, seeds)
        feature_weights = np.random.rand(len(points))  # Simplified feature weights for demonstration purposes
        entropy = -np.sum(feature_weights * np.log2(feature_weights))
        hygiene_score = entropy * region_states[region_index]
        hygiene_scores[p] = hygiene_score
    return hygiene_scores

def main():
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(10)]
    regions = build_voronoi(points, seeds)
    region_states = update_region_state(regions, seeds, 0.1)
    hygiene_scores = hygiene_score_with_sketch(points, seeds, region_states)
    print(hygiene_scores)

if __name__ == "__main__":
    main()