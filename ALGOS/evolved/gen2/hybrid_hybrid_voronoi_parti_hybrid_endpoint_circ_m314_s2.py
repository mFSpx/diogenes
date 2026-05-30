# DARWIN HAMMER — match 314, survivor 2
# gen: 2
# parent_a: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py (gen1)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# born: 2026-05-29T23:28:15Z

"""Hybrid Voronoi‑Thermal‑Morphology Circuit Breaker.

Parent A: Voronoi partition + Schoolfield temperature‑activity model.
Parent B: EndpointCircuitBreaker + Morphology (sphericity, flatness).

Mathematical bridge:
* Each Voronoi seed is enriched with a Morphology and a local temperature (°C).
* The Schoolfield model maps that temperature to a normalized activity 𝛼∈[0,1].
* The circuit‑breaker failure threshold τ for the region is scaled by the activity
  and by the seed’s sphericity σ (geometric shape factor):

      τ = max(1, int(base_threshold * (1 − α) * σ))

Thus a hot, highly active region (α≈1) yields a low tolerance for failures,
while a cool, low‑activity region (α≈0) tolerates more failures. The Voronoi
partition supplies the spatial grouping, the thermal model supplies α, and the
morphology supplies σ – a single unified system.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Voronoi assignment of *points* to the nearest *seeds*."""
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15  # K
    t_high: float = 307.15  # K
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield model for temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    """Map a temperature (°C) to a 0‑1 activity gate using the Schoolfield curve."""
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    # Compute the maximum rate over the sampling interval once (could be cached)
    max_rate = max(
        developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params)
        for i in range(samples)
    )
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric mean of dimensions divided by the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Ratio of the shortest dimension to the geometric mean."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return min(length, width, height) / gm


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """True if the circuit is closed (endpoint usable)."""
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
        }


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    """A Voronoi seed enriched with morphology and a local temperature."""
    position: Tuple[float, float]
    morphology: Morphology
    temperature_c: float  # ambient temperature at the seed location


# ----------------------------------------------------------------------
# Hybrid operations (the three required functions)
# ----------------------------------------------------------------------


def assign_regions(points: List[Tuple[float, float]], seeds: List[Seed]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Perform Voronoi assignment of *points* to the nearest *seeds*.

    Returns a mapping ``region_index -> list_of_points``.
    """
    seed_positions = [s.position for s in seeds]
    return assign(points, seed_positions)


def region_activity(seeds: List[Seed], region_points: List[Tuple[float, float]]) -> float:
    """
    Compute the normalized thermal activity for a region.

    The activity is the average temperature of the region's points (°C) fed
    through the Schoolfield normalisation. If a region has no points, the seed's
    own temperature is used.
    """
    if not region_points:
        avg_temp = seeds[0].temperature_c  # fallback (won't be used in practice)
    else:
        # For demonstration we treat the x‑coordinate as a proxy for temperature.
        # In a real system each point would carry its own temperature.
        temps = [p[0] for p in region_points]  # simplistic proxy
        avg_temp = sum(temps) / len(temps)
    # Blend seed temperature with region average to smooth extremes
    blended = 0.5 * avg_temp + 0.5 * seeds[0].temperature_c
    return normalized_activity(blended)


def configure_breakers(
    seeds: List[Seed],
    regions: Dict[int, List[Tuple[float, float]]],
    base_threshold: int = 3,
) -> List[EndpointCircuitBreaker]:
    """
    Create one ``EndpointCircuitBreaker`` per region.

    The failure threshold τ for a region is modulated by:
        τ = max(1, int(base_threshold * (1 - α) * σ))

    where α is the region's normalized activity and σ is the seed's sphericity.
    """
    breakers: List[EndpointCircuitBreaker] = []
    for idx, seed in enumerate(seeds):
        pts = regions.get(idx, [])
        α = region_activity([seed], pts)
        σ = sphericity_index(seed.morphology.length, seed.morphology.width, seed.morphology.height)
        threshold = max(1, int(base_threshold * (1.0 - α) * σ))
        breakers.append(EndpointCircuitBreaker(failure_threshold=threshold))
    return breakers


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo() -> None:
    # Generate synthetic seeds with random geometry and temperature
    random.seed(42)
    seeds: List[Seed] = []
    for i in range(5):
        pos = (random.uniform(0, 100), random.uniform(0, 100))
        morph = Morphology(
            length=random.uniform(0.5, 5.0),
            width=random.uniform(0.5, 5.0),
            height=random.uniform(0.5, 5.0),
            mass=random.uniform(1.0, 10.0),
        )
        temp_c = random.uniform(10.0, 35.0)  # plausible operating temps
        seeds.append(Seed(position=pos, morphology=morph, temperature_c=temp_c))

    # Generate random points (x used as temperature proxy)
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]

    # Hybrid pipeline
    regions = assign_regions(points, seeds)
    breakers = configure_breakers(seeds, regions, base_threshold=4)

    # Simulate a few random events per region to show circuit‑breaker behaviour
    for idx, breaker in enumerate(breakers):
        # Randomly decide success/failure based on region activity
        α = region_activity([seeds[idx]], regions.get(idx, []))
        for _ in range(6):
            if random.random() < α:  # higher activity → higher chance of failure
                breaker.record_failure()
            else:
                breaker.record_success()
        print(f"Region {idx}:")
        print(f"  Seed position: {seeds[idx].position}")
        print(f"  Temperature (°C): {seeds[idx].temperature_c:.2f}")
        print(f"  Sphericity σ: {sphericity_index(seeds[idx].morphology.length, seeds[idx].morphology.width, seeds[idx].morphology.height):.3f}")
        print(f"  Normalized activity α: {α:.3f}")
        print(f"  Configured threshold: {breaker.failure_threshold}")
        print(f"  Final state: {breaker.as_dict()}")
        print()


if __name__ == "__main__":
    _demo()