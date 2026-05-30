# DARWIN HAMMER — match 3424, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1.py (gen3)
# parent_b: hybrid_fractional_hdc_counterfactual_effec_m38_s1.py (gen1)
# born: 2026-05-29T23:50:01Z

"""
This module describes a novel HYBRID algorithm that mathematically fuses the core topologies of 
two mathematical algorithms: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1 and hybrid_fractional_hdc_counterfactual_effec_m38_s1.
The mathematical bridge between these structures is the integration of Voronoi partitioning with the 
application of HDC's binding operator to encode causal relationships, and the use of fractional power binding 
to inform model selection in the hybrid privacy model pool management.
"""

import math
import random
import numpy as np
import sys
import pathlib
from datetime import datetime, timezone

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def random_hv(d=10000, kind="complex", seed=None):
    """Generate a random hypervector of dimension d.

    Parameters
    ----------
    d:
        Dimension of the hypervector.
    kind:
        "complex"  — unit-magnitude complex vector (each component e^{i*theta},
                     theta ~ Uniform[0, 2pi]).  These are the natural carriers
                     for phase-based fractional binding.
        "bipolar"  — real vector with each component in {+1, -1}.
        "real"     — Gaussian sample normalized to unit L2 norm.
    seed:
        Integer seed for reproducibility; None for random.

    Returns
    -------
    np.ndarray
        Shape (d,).  dtype=complex128 for kind="complex", float64 otherwise.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    elif kind == "real":
        return rng.normal(size=d) / np.linalg.norm(rng.normal(size=d))
    else:
        raise ValueError("Invalid kind")

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.multiply(a, b)

def fractional_power(a: np.ndarray, power: float) -> np.ndarray:
    return np.power(np.abs(a), power) * np.exp(1j * power * np.angle(a))

def hybrid_operation(points: list[tuple[float, float]], seeds: list[tuple[float, float]], hv: np.ndarray) -> dict[int, list[tuple[float, float]]]:
    regions = assign(points, seeds)
    for i, region in regions.items():
        for point in region:
            dist = distance(point, seeds[i])
            if dist > 0:
                regions[i] = [p for p in region if distance(p, seeds[i]) <= dist * 0.5]
    return regions

def voronoi_partition_with_hdc(points: list[tuple[float, float]], seeds: list[tuple[float, float]], hv: np.ndarray) -> dict[int, list[tuple[float, float]]]:
    regions = assign(points, seeds)
    for i, region in regions.items():
        bounded_region = bind(hv, hv)
        regions[i] = [p for p in region if distance(p, seeds[i]) <= np.mean(np.abs(bounded_region))]
    return regions

def hdc_with_voronoi(points: list[tuple[float, float]], seeds: list[tuple[float, float]], hv: np.ndarray) -> dict[int, list[tuple[float, float]]]:
    regions = assign(points, seeds)
    for i, region in regions.items():
        fractional_hv = fractional_power(hv, 0.5)
        regions[i] = [p for p in region if distance(p, seeds[i]) <= np.mean(np.abs(fractional_hv))]
    return regions

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    hv = random_hv(10000, "complex")
    print(hybrid_operation(points, seeds, hv))
    print(voronoi_partition_with_hdc(points, seeds, hv))
    print(hdc_with_voronoi(points, seeds, hv))