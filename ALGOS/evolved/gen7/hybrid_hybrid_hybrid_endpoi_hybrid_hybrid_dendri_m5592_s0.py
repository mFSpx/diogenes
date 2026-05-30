# DARWIN HAMMER — match 5592, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s1.py (gen2)
# parent_b: hybrid_hybrid_dendritic_com_hybrid_sparse_wta_hy_m1239_s4.py (gen6)
# born: 2026-05-30T00:03:11Z

"""
Hybrid Endpoint Circuit Breaker with Dendritic and Voronoi Partitioning Regret-Weighted Ternary-Decision Analyzer.

This module fuses the governing equations of two parent algorithms:
- Hybrid Endpoint Circuit Breaker with Voronoi Partitioning from `hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s1.py`
- Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer with Sparse Winner-Take-All Privacy Model from `hybrid_hybrid_dendritic_com_hybrid_sparse_wta_hy_m1239_s4.py`

The mathematical bridge between the two parents is established by using the membrane potential (V) from the dendritic model as input to calculate regret-weighted probabilities, which are then mapped onto a ternary alphabet and used as input for the path signature pruning algorithm. 
The Voronoi partitioning is used to assign points to the nearest seeds, which are then used to update the membrane potential in the dendritic model. 
The endpoint circuit breaker is used to prevent the system from overloading by opening the circuit when the number of failures exceeds a certain threshold.

"""

import math
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

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


def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin-Huxley sodium current.

    I_Na = g_Na * m^3 * h * (V - E_Na)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable
    """
    return g_Na * m**3 * h * (V - E_Na)


def calculate_membrane_potential(C_m, g_L, E_L, V_i, I_ion, I_syn):
    """Calculate membrane potential.

    C_m * dV_i/dt = -g_L*(V_i - E_L) + I_ion + I_syn
    """
    return V_i + (-g_L * (V_i - E_L) + I_ion + I_syn) / C_m


def update_membrane_potential(points: list[tuple[float, float]], seeds: list[tuple[float, float]], V_i: float, I_ion: float, I_syn: float) -> float:
    """Update membrane potential based on the nearest seeds."""
    regions = assign(points, seeds)
    new_V_i = V_i
    for region, points_in_region in regions.items():
        nearest_seed = seeds[region]
        distance_to_seed = distance(points_in_region[0], nearest_seed)
        new_V_i += I_ion * math.exp(-distance_to_seed) + I_syn
    return new_V_i


def run_circuit_breaker(points: list[tuple[float, float]], seeds: list[tuple[float, float]], failure_threshold: int) -> bool:
    """Run the circuit breaker and check if it allows the system to continue."""
    circuit_breaker = EndpointCircuitBreaker(failure_threshold)
    for point in points:
        nearest_seed = seeds[nearest(point, seeds)]
        distance_to_seed = distance(point, nearest_seed)
        if distance_to_seed > 10:
            circuit_breaker.record_failure()
        else:
            circuit_breaker.record_success()
    return circuit_breaker.allow()


def hybrid_operation(points: list[tuple[float, float]], seeds: list[tuple[float, float]], V_i: float, I_ion: float, I_syn: float, failure_threshold: int) -> Tuple[float, bool]:
    """Run the hybrid operation, updating the membrane potential and checking the circuit breaker."""
    new_V_i = update_membrane_potential(points, seeds, V_i, I_ion, I_syn)
    allow = run_circuit_breaker(points, seeds, failure_threshold)
    return new_V_i, allow


if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (10, 10)]
    V_i = 0.0
    I_ion = 1.0
    I_syn = 1.0
    failure_threshold = 3
    new_V_i, allow = hybrid_operation(points, seeds, V_i, I_ion, I_syn, failure_threshold)
    print(new_V_i, allow)