# DARWIN HAMMER — match 110, survivor 3
# gen: 4
# parent_a: hybrid_distributed_leader_e_thanatosis_m65_s2.py (gen1)
# parent_b: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py (gen3)
# born: 2026-05-29T23:26:52Z

"""Hybrid Algorithm Fusing Distributed Leader Election via Simulated Annealing and Physarum Network Dynamics.

This module integrates the core topologies of `hybrid_distributed_leader_e_thanatosis_m65_s2.py` (Parent A) and 
`hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py` (Parent B) by recognizing that both can be 
represented as interacting dynamical systems.

The mathematical bridge between the two parents is established through the interpretation of the 
broadcast probability in Parent A as a pressure difference in Parent B, driving a flux through the network.
The conductance in Parent B is updated based on the flux, which in turn modulates the propensity of 
the bandit policy in Parent B. The hybrid algorithm embeds the simulated-annealing process of Parent A 
into the Physarum network dynamics of Parent B.

The governing equations of both parents are integrated through the following interface:
- The broadcast probability `p_phase` in Parent A is used to compute the pressure difference 
  `Δp = p_up – p_down` in Parent B, driving the flux `q = 𝒞 / ℓ * Δp`.
- The conductance `𝒞` in Parent B is updated based on the flux `q`, which feeds back into the 
  bandit policy, modulating its propensities.

The hybrid algorithm exposes three core functions: `hybrid_flux`, `update_hybrid_conductance`, 
and `hybrid_bandit_step`. A lightweight `HybridNetwork` class orchestrates actions, stores, 
and the conductance matrix.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

Node = Hashable
Graph = Mapping[Node, Set[Node]]

@dataclass
class HybridNetwork:
    conductance: Dict[Tuple[Node, Node], float]
    edge_length: Dict[Tuple[Node, Node], float]
    pressure: Dict[Node, float]

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

def hybrid_temperature(phases: int, phase: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    p = broadcast_probability(phases, phase)
    T_phase = cooling_temperature(phase-1, t0, alpha)
    return T_phase * p

def hybrid_flux(network: HybridNetwork, node_a: Node, node_b: Node) -> float:
    conductance = network.conductance.get((node_a, node_b), 0.0)
    edge_length = network.edge_length.get((node_a, node_b), 1.0)
    pressure_a = network.pressure.get(node_a, 0.0)
    pressure_b = network.pressure.get(node_b, 0.0)
    return conductance / edge_length * (pressure_a - pressure_b)

def update_hybrid_conductance(network: HybridNetwork, node_a: Node, node_b: Node, 
                              q: float, gain: float = 1.0, decay: float = 0.1, 
                              dt: float = 0.01) -> None:
    conductance = network.conductance.get((node_a, node_b), 0.0)
    network.conductance[(node_a, node_b)] = max(0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_bandit_step(network: HybridNetwork, node: Node, phases: int, phase: int, 
                       t0: float = 1.0, alpha: float = 0.95) -> Optional[Node]:
    temperature = hybrid_temperature(phases, phase, t0, alpha)
    neighbors = network.pressure.keys()
    best_neighbor = max(neighbors, key=lambda n: network.pressure[n])
    q = hybrid_flux(network, node, best_neighbor)
    update_hybrid_conductance(network, node, best_neighbor, q)
    if random.random() < math.exp(-q / temperature):
        return best_neighbor
    return None

if __name__ == "__main__":
    network = HybridNetwork(
        conductance={(0, 1): 1.0, (1, 2): 1.0},
        edge_length={(0, 1): 1.0, (1, 2): 1.0},
        pressure={0: 1.0, 1: 0.5, 2: 0.0}
    )
    print(hybrid_flux(network, 0, 1))
    update_hybrid_conductance(network, 0, 1, 0.5)
    print(network.conductance)
    print(hybrid_bandit_step(network, 0, 10, 5))