# DARWIN HAMMER — match 4403, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m2732_s1.py (gen4)
# born: 2026-05-29T23:55:25Z

"""
Hybrid algorithm fusing the leader election dynamics of `hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s0.py` 
with the NLMS-Bandit-Pheromone System of `hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m2732_s1.py`.

The mathematical bridge is established by interpreting the broadcast probability 
as a pheromone signal driving the NLMS weight update. The conductance 
evolves according to the absolute flux, and the updated conductance modulates 
the NLMS step-size μ (high conductance → larger μ, low conductance → smaller μ).
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

Node = str
Graph = Dict[Node, List[Node]]

@dataclass
class HybridLeaderElection:
    graph: Graph
    phases: int
    phase: int
    t0: float = 1.0
    alpha: float = 0.95
    edge_length: float = 1.0
    eps: float = 1e-12
    conductance: Dict[Tuple[Node, Node], float] = None

    def __post_init__(self):
        if self.conductance is None:
            self.conductance = {tuple(edge): 1.0 for node in self.graph for edge in [(node, neighbor) for neighbor in self.graph[node]]}

    def broadcast_probability(self) -> float:
        """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
        if self.phases < 1 or self.phase < 1:
            raise ValueError("phases and phase must be positive")
        return min(1.0, 1.0 / (2 ** max(0, self.phases - self.phase)))

    def cooling_temperature(self) -> float:
        """Original B: exponential cooling schedule."""
        if self.phase < 0 or self.t0 < 0 or not (0 <= self.alpha <= 1):
            raise ValueError("invalid cooling parameters")
        return self.t0 * (self.alpha ** (self.phase - 1))

    def hybrid_temperature(self) -> float:
        """Combine the decay of broadcast probability and annealing temperature."""
        p = self.broadcast_probability()
        T = self.cooling_temperature()
        return T * p

    def flux(self, node_a: Node, node_b: Node, pressure_a: float, pressure_b: float) -> float:
        """Physarum-style flux calculation."""
        conductance = self.conductance.get((node_a, node_b), 1.0)
        return conductance * (pressure_a - pressure_b)

    def update_conductance(self, node_a: Node, node_b: Node, flux: float) -> None:
        """Update conductance based on absolute flux."""
        conductance = self.conductance.get((node_a, node_b), 1.0)
        self.conductance[(node_a, node_b)] = conductance + abs(flux)

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str = ""          # optional semantic label
    score: float = 0.0       # confidence from extractor
    backend: str = "hybrid"

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = w·x."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Classic Normalised Least‑Mean‑Squares adaptation.

    Returns (new_weights, error) where error = target - 
    """
    prediction = nlms_predict(weights, x)
    error = target - prediction
    weights_new = weights + mu * error * x / (np.dot(x, x) + eps)
    return weights_new, error

def hybrid_nlms_bandit_physarum(
    hybrid_leader_election: HybridLeaderElection,
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    node_a: Node,
    node_b: Node,
    pressure_a: float,
    pressure_b: float,
) -> Tuple[np.ndarray, float, float]:
    """Hybrid NLMS-Bandit-Pheromone-Physarum System."""
    flux = hybrid_leader_election.flux(node_a, node_b, pressure_a, pressure_b)
    conductance = hybrid_leader_election.conductance.get((node_a, node_b), 1.0)
    mu = 0.5 * conductance
    weights_new, error = nlms_update(weights, x, target, mu)
    hybrid_leader_election.update_conductance(node_a, node_b, flux)
    return weights_new, error, flux

if __name__ == "__main__":
    graph = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
    hybrid_leader_election = HybridLeaderElection(graph, phases=10, phase=5)
    weights = np.array([0.1, 0.2, 0.3])
    x = np.array([1.0, 2.0, 3.0])
    target = 10.0
    node_a = "A"
    node_b = "B"
    pressure_a = 1.0
    pressure_b = 0.5
    weights_new, error, flux = hybrid_nlms_bandit_physarum(hybrid_leader_election, weights, x, target, node_a, node_b, pressure_a, pressure_b)
    print("Weights new:", weights_new)
    print("Error:", error)
    print("Flux:", flux)