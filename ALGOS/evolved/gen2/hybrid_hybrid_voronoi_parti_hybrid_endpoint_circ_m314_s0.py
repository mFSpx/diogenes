# DARWIN HAMMER — match 314, survivor 0
# gen: 2
# parent_a: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py (gen1)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# born: 2026-05-29T23:28:15Z

"""
This module fuses the mathematical structures of the 'hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0' 
and 'hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3' algorithms. The bridge between the two 
structures lies in the assignment of points to regions based on their distances to seeds, and the 
circuit-breaker mechanism that opens or closes based on the number of failures. In the hybrid system, 
we integrate the developmental rate calculation from the first algorithm with the circuit-breaker 
mechanism from the second algorithm. The mathematical interface is formed by using the circuit-breaker 
state to gate the assignment of points to thermal regions.

Authors: based on 'hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0' and 
         'hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3'
"""

import math
import numpy as np
import random
import sys
from pathlib import Path

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

class SchoolfieldParams:
    def __init__(self, rho_25: float = 1.0, delta_h_activation: float = 12_000.0, 
                 t_low: float = 283.15, t_high: float = 307.15, delta_h_low: float = -45_000.0, 
                 delta_h_high: float = 65_000.0, r_cal: float = 1.987):
        self.rho_25 = rho_25
        self.delta_h_activation = delta_h_activation
        self.t_low = t_low
        self.t_high = t_high
        self.delta_h_low = delta_h_low
        self.delta_h_high = delta_h_high
        self.r_cal = r_cal

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def assign_thermal_region(point: tuple[float, float], seeds: list[tuple[float, float]], 
                          params: SchoolfieldParams, circuit_breaker: EndpointCircuitBreaker) -> int:
    if not circuit_breaker.allow():
        raise ValueError("Circuit breaker is open")
    return nearest(point, seeds)

def calculate_developmental_rate_at_point(point: tuple[float, float], seeds: list[tuple[float, float]], 
                                          params: SchoolfieldParams, circuit_breaker: EndpointCircuitBreaker) -> float:
    region = assign_thermal_region(point, seeds, params, circuit_breaker)
    return developmental_rate(c_to_k(seeds[region][0]), params)

def calculate_average_developmental_rate(points: list[tuple[float, float]], seeds: list[tuple[float, float]], 
                                         params: SchoolfieldParams, circuit_breaker: EndpointCircuitBreaker) -> float:
    total_rate = 0.0
    for point in points:
        total_rate += calculate_developmental_rate_at_point(point, seeds, params, circuit_breaker)
    return total_rate / len(points)

if __name__ == "__main__":
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    seeds = [(0.5, 0.5), (1.5, 1.5)]
    params = SchoolfieldParams()
    circuit_breaker = EndpointCircuitBreaker()
    circuit_breaker.record_success()
    average_rate = calculate_average_developmental_rate(points, seeds, params, circuit_breaker)
    print(average_rate)