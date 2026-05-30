# DARWIN HAMMER — match 314, survivor 1
# gen: 2
# parent_a: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py (gen1)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# born: 2026-05-29T23:28:15Z

"""
Hybrid of voronoi_partition_poikilotherm_schoolf_m49_s0.py and hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py.
The mathematical bridge between the two parents lies in the concept of 'regions' and 'states'. 
In the Voronoi partition, regions are defined by proximity to seeds. 
In the endpoint circuit breaker, states are defined by the circuit's openness or closeness. 
The hybrid algorithm combines these concepts by assigning a 'state' to each Voronoi region, 
based on the circuit breaker's state at the region's seed point.

"""

import math
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

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
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
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
        return not self.open

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def hybrid_operation(points: list[tuple[float, float]], seeds: list[tuple[float, float]], 
                      circuit_breaker: EndpointCircuitBreaker, morphology: Morphology) -> dict[int, dict[str, Any]]:
    regions = assign(points, seeds)
    hybrid_regions = {}
    for seed_idx, region_points in regions.items():
        seed_state = circuit_breaker.allow()
        region_state = {
            'circuit_state': seed_state,
            'sphericity_index': sphericity_index(morphology.length, morphology.width, morphology.height),
            'normalized_activity': normalized_activity(20.0)  # example temperature
        }
        hybrid_regions[seed_idx] = region_state
    return hybrid_regions

def get_hybrid_region_states(points: list[tuple[float, float]], seeds: list[tuple[float, float]], 
                             circuit_breaker: EndpointCircuitBreaker, morphology: Morphology) -> list[dict[str, Any]]:
    hybrid_regions = hybrid_operation(points, seeds, circuit_breaker, morphology)
    region_states = []
    for seed_idx, region_state in hybrid_regions.items():
        region_states.append({
            'seed_idx': seed_idx,
            'region_state': region_state
        })
    return region_states

if __name__ == "__main__":
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    seeds = [(0.0, 0.0), (2.0, 2.0)]
    circuit_breaker = EndpointCircuitBreaker()
    morphology = Morphology(1.0, 1.0, 1.0, 1.0)
    region_states = get_hybrid_region_states(points, seeds, circuit_breaker, morphology)
    print(region_states)