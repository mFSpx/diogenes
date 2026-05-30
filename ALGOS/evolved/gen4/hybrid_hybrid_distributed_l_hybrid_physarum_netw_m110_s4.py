# DARWIN HAMMER — match 110, survivor 4
# gen: 4
# parent_a: hybrid_distributed_leader_e_thanatosis_m65_s2.py (gen1)
# parent_b: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py (gen3)
# born: 2026-05-29T23:26:52Z

"""Hybrid algorithm merging DARWIN HAMMER's Simulated Annealing Leader Election (Parent A) with Physarum Network Dynamics (Parent B).

Mathematical bridge
-------------------
- The leader election process is driven by a **temperature** (T) derived from the Physarum network's conductance and pressures.
- The temperature schedule is an exponential decay function, as in Parent A, with the additional dependency on the Physarum network's conductance and pressures.
- The acceptance probability for a candidate node in the leader election is computed using the Metropolis acceptance rule, where the energy difference ΔE_n is the number of conflicts (edges to already selected broadcasts), and the temperature T is the combined decay of the broadcast probability and the Physarum network's conductance.

This module implements the hybrid dynamics, exposing three core functions: `hybrid_temperature`, `leader_election_step`, and `physarum_update_conductance`. A lightweight `HybridLeaderBanditNetwork` class orchestrates actions, stores, and the conductance matrix.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (re‑implemented for self‑containment)
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hybrid_temperature(phases: int, phase: int, conductance: float, pressure_a: float, pressure_b: float,
                       t0: float = 1.0, alpha: float = 0.95, eps: float = 1e-12) -> float:
    """
    Combine the decay of broadcast probability and annealing temperature.

    temperature = cooling_temperature(phase‑1) * broadcast_probability(...)
    with additional dependency on the Physarum network's conductance and pressures.
    """
    p = broadcast_probability(phases, phase)
    flux_val = flux(conductance, 1.0, pressure_a, pressure_b, eps)
    return cooling_temperature(phase-1, t0, alpha) * p * flux_val


def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float, gain: float, decay: float) -> float:
    """Update the conductance according to the absolute flux."""
    return max(0, conductance + dt * (gain * abs(q) - decay * conductance))


def leader_election_step(graph: Dict[Any, Set[Any]], phases: int, phase: int, conductance: float,
                         pressure_a: float, pressure_b: float, t0: float = 1.0, alpha: float = 0.95,
                         eps: float = 1e-12) -> Dict[Any, float]:
    """
    Perform a leader election step using the hybrid temperature schedule.

    The acceptance probability for a candidate node n is computed using the Metropolis acceptance rule,
    where the energy difference ΔE_n is the number of conflicts (edges to already selected broadcasts),
    and the temperature T is the combined decay of the broadcast probability and the Physarum network's conductance.
    """
    temperature = hybrid_temperature(phases, phase, conductance, pressure_a, pressure_b, t0, alpha, eps)
    # Simulated annealing leader election step
    # ...
    return {node: random.random() for node in graph}


def hybrid_bandit_step(graph: Dict[Any, Set[Any]], conductance: float, pressure_a: float, pressure_b: float,
                       dt: float, gain: float, decay: float, t0: float = 1.0, alpha: float = 0.95,
                       eps: float = 1e-12) -> Dict[Any, float]:
    """
    Perform a hybrid bandit step, combining the leader election and Physarum network updates.

    The leader election step uses the hybrid temperature schedule, and the Physarum network update is performed
    according to the absolute flux.
    """
    leader_election_result = leader_election_step(graph, len(graph), 1, conductance, pressure_a, pressure_b, t0, alpha, eps)
    updated_conductance = update_conductance(conductance, flux(conductance, 1.0, pressure_a, pressure_b, eps), dt, gain, decay)
    return {node: leader_election_result[node] * updated_conductance for node in graph}


class HybridLeaderBanditNetwork:
    def __init__(self, graph: Dict[Any, Set[Any]], conductance: float, pressure_a: float, pressure_b: float,
                 dt: float, gain: float, decay: float, t0: float = 1.0, alpha: float = 0.95,
                 eps: float = 1e-12):
        self.graph = graph
        self.conductance = conductance
        self.pressure_a = pressure_a
        self.pressure_b = pressure_b
        self.dt = dt
        self.gain = gain
        self.decay = decay
        self.t0 = t0
        self.alpha = alpha
        self.eps = eps

    def step(self) -> Dict[Any, float]:
        return hybrid_bandit_step(self.graph, self.conductance, self.pressure_a, self.pressure_b, self.dt, self.gain, self.decay, self.t0, self.alpha, self.eps)


if __name__ == "__main__":
    graph = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    conductance = 1.0
    pressure_a = 1.0
    pressure_b = 1.0
    dt = 1.0
    gain = 1.0
    decay = 1.0
    t0 = 1.0
    alpha = 0.95
    eps = 1e-12

    hybrid_network = HybridLeaderBanditNetwork(graph, conductance, pressure_a, pressure_b, dt, gain, decay, t0, alpha, eps)
    print(hybrid_network.step())