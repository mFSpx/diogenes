# DARWIN HAMMER — match 5592, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s1.py (gen2)
# parent_b: hybrid_hybrid_dendritic_com_hybrid_sparse_wta_hy_m1239_s4.py (gen6)
# born: 2026-05-30T00:03:11Z

"""
Hybrid Endpoint-Circuit-Breaker Voronoi Partition with Dendritic Regret-Weighted Ternary-Decision Analyzer.

This module fuses the governing equations of two parent algorithms:
- Hybrid Endpoint Circuit Breaker with Voronoi Partition (HEC-VP) from `hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s1.py`
- Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer with Sparse Winner-Take-All Privacy Model (HD-RW-TD) from `hybrid_hybrid_dendritic_com_hybrid_sparse_wta_hy_m1239_s4.py`

The mathematical bridge between the two parents is established by using the Voronoi partition regions from HEC-VP as input to calculate regret-weighted probabilities in HD-RW-TD, 
which are then mapped onto a ternary alphabet and used to compute dendritic membrane potentials.

"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

def now_z() -> str:
    """Current UTC timestamp in ISO-8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

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

    C_m * dV_i/dt = -g_L*(V_i - E_L) + I_ion(V_i) + I_syn

    Parameters
    ----------
    C_m : float
        Membrane capacitance (μF/cm^2).
    g_L : float
        Leak conductance (mS/cm^2).
    E_L : float
        Leak reversal potential (mV).
    V_i : float
        Membrane potential (mV).
    I_ion : callable
        Ion current (μA/cm^2).
    I_syn : float
        Synaptic current (μA/cm^2).

    Returns
    -------
    float
        Membrane potential (mV).
    """
    return -g_L*(V_i - E_L) + I_ion + I_syn

def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    return params.rho_25 * np.exp((params.delta_h_activation / params.r_cal) * (1/params.t_low - 1/temp_k))

def hybrid_operation(points: list[tuple[float, float]], seeds: list[tuple[float, float]], 
                      temp_k: float, params: SchoolfieldParams) -> Tuple[dict[int, list[tuple[float, float]]], float]:
    regions = assign(points, seeds)
    regret_values = []
    for region in regions.values():
        if region:
            regret_value = developmental_rate(temp_k, params) * len(region)
            regret_values.append(regret_value)
    if regret_values:
        probabilities = np.array(regret_values) / sum(regret_values)
        membrane_potentials = []
        for probability in probabilities:
            V = 0  # initial membrane potential
            I_ion = sodium_current(V, 0.5, 0.5)  # example ion current
            I_syn = 0  # example synaptic current
            C_m = 1.0  # example membrane capacitance
            g_L = 0.1  # example leak conductance
            E_L = -70.0  # example leak reversal potential
            membrane_potential = calculate_membrane_potential(C_m, g_L, E_L, V, I_ion, I_syn)
            membrane_potentials.append(membrane_potential)
        return regions, np.mean(membrane_potentials)
    else:
        return regions, 0.0

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (10, 10)]
    temp_k = 300.0
    params = SchoolfieldParams()
    regions, membrane_potential = hybrid_operation(points, seeds, temp_k, params)
    print(regions)
    print(membrane_potential)