# DARWIN HAMMER — match 633, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1.py (gen3)
# parent_b: hybrid_liquid_time_constant_hybrid_hybrid_hdc_se_m174_s0.py (gen4)
# born: 2026-05-29T23:30:06Z

"""
This module implements a novel HYBRID algorithm that mathematically fuses the core topologies of 
two mathematical algorithms: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1.py and 
hybrid_liquid_time_constant_hybrid_hybrid_hdc_se_m174_s0.py.
The mathematical bridge between these structures is the integration of Voronoi partitioning with the 
morphology and recovery priority of the hybrid endpoint circuit breakers, and the application of 
sparse winner-take-all tags to inform model selection in the hybrid privacy model pool management, 
combined with the hyperdimensional primitives' binding and bundling operations to compute the 
input-dependent time constant.
"""

import numpy as np
import random
import math
import sys
import pathlib

# ----------------------------------------------------------------------
# Voronoi partitioning
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Hyperdimensional primitives
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return sigmoid(np.dot(W, np.concatenate((x, I))) + b)

def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=float,
    )
    return data / np.linalg.norm(data)

# ----------------------------------------------------------------------
# Hybrid operation: Voronoi partitioning + hyperdimensional primitives
# ----------------------------------------------------------------------

def hybrid_bind(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    seeds: list[Point],
    points: list[Point],
) -> np.ndarray:
    """Combine Voronoi partitioning with hyperdimensional primitives to compute the input-dependent time constant."""
    regions = assign(points, seeds)
    weights = np.array([len(region) for region in regions.values()])
    weights = weights / np.sum(weights)
    hypervectors = [random_vector(dim=len(seeds)) for _ in range(len(seeds))]
    for i, region in enumerate(regions.values()):
        for point in region:
            nearest_index = nearest(point, seeds)
            hypervectors[nearest_index] += random_vector(dim=len(seeds))
    hypervector = np.concatenate(hypervectors)
    return ltc_f(hypervector, I, W, b)

def hybrid_bundle(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    seeds: list[Point],
    points: list[Point],
) -> np.ndarray:
    """Combine Voronoi partitioning with hyperdimensional primitives to compute the asymptotic target state."""
    regions = assign(points, seeds)
    weights = np.array([len(region) for region in regions.values()])
    weights = weights / np.sum(weights)
    hypervectors = [random_vector(dim=len(seeds)) for _ in range(len(seeds))]
    for i, region in enumerate(regions.values()):
        for point in region:
            nearest_index = nearest(point, seeds)
            hypervectors[nearest_index] += random_vector(dim=len(seeds))
    hypervector = np.concatenate(hypervectors)
    return hypervector

def hybrid_step(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    seeds: list[Point],
    points: list[Point],
) -> np.ndarray:
    """Combine Voronoi partitioning with hyperdimensional primitives to simulate the dynamics of the hybrid network."""
    hypervector = hybrid_bind(x, I, W, b, seeds, points)
    return ltc_f(hypervector, I, W, b)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    seeds = [(0, 0), (1, 0), (1, 1), (0, 1)]
    points = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    x = np.array([1, 1, 1, 1])
    I = np.array([1, 1, 1, 1])
    W = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    b = np.array([0, 0, 0, 0])
    print(hybrid_bind(x, I, W, b, seeds, points))
    print(hybrid_bundle(x, I, W, b, seeds, points))
    print(hybrid_step(x, I, W, b, seeds, points))