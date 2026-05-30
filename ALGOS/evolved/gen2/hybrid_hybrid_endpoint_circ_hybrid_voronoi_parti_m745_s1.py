# DARWIN HAMMER — match 745, survivor 1
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s2.py (gen1)
# parent_b: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py (gen1)
# born: 2026-05-29T23:30:47Z

import math
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

def now_z() -> str:
    """Current UTC timestamp in ISO-8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open
        }


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


def calculate_health_score(circuit_breaker: EndpointCircuitBreaker, recovery_priority: float) -> float:
    reliability_term = 1.0 - (circuit_breaker.failures / circuit_breaker.failure_threshold)
    return reliability_term * recovery_priority


def weighted_voronoi_partition(points: list[tuple[float, float]], seeds: list[tuple[float, float]], health_scores: list[float]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p, score in zip(points, health_scores):
        nearest_seed = nearest(p, seeds)
        regions[nearest_seed].append(p)
    return regions


def distance_weighted_voronoi_partition(points: list[tuple[float, float]], seeds: list[tuple[float, float]], health_scores: list[float]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p, score in zip(points, health_scores):
        distances = [distance(p, seed) for seed in seeds]
        nearest_seed = min(range(len(seeds)), key=lambda i: (distances[i] / (1 + health_scores[i]), i))
        regions[nearest_seed].append(p)
    return regions


def assign_thermal_region(points: list[tuple[float, float]], seeds: list[tuple[float, float]], circuit_breakers: list[EndpointCircuitBreaker], recovery_priorities: list[float]) -> dict[int, list[tuple[float, float]]]:
    health_scores = [calculate_health_score(c, p) for c, p in zip(circuit_breakers, recovery_priorities)]
    return distance_weighted_voronoi_partition(points, seeds, health_scores)


if __name__ == "__main__":
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(10)]
    seeds = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(3)]
    circuit_breakers = [EndpointCircuitBreaker() for _ in range(len(points))]
    recovery_priorities = [random.uniform(0, 1) for _ in range(len(points))]
    for c in circuit_breakers:
        c.record_success()
    regions = assign_thermal_region(points, seeds, circuit_breakers, recovery_priorities)
    for seed, region in regions.items():
        print(f"Seed {seed}: {region}")