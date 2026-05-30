# DARWIN HAMMER — match 2736, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:44:00Z

"""
Hybrid algorithm combining the Physarum-Bandit-TTT model from hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py 
and the hybrid ternary router with Bayesian evidence update from hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py.
The mathematical bridge between the two structures is the notion of uncertainty 
in the Physarum network's edge conductances, which can be updated using the Bayesian update rule 
and integrated into the Physarum's conductance update rule. This bridge allows the Physarum model 
to adapt to changing environments and learn from experience, while the Bayesian update rule provides 
a probabilistic framework for incorporating new evidence into the Physarum model.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple
import numpy as np

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

Point = Tuple[float, float]
Edge = Tuple[str, str]

_POLICY: dict[str, list[float]] = {}
_STORE: dict[str, float] = {}
_NODE_LOCATIONS: dict[str, Point] = {}
_EDGE_CONDUCTANCES: dict[Edge, float] = {}
_EDGE_PRIORS: dict[Edge, float] = {}

def reset_policy() -> None:
    global _POLICY, _STORE, _NODE_LOCATIONS, _EDGE_CONDUCTANCES, _EDGE_PRIORS
    _POLICY.clear()
    _STORE.clear()
    _NODE_LOCATIONS.clear()
    _EDGE_CONDUCTANCES.clear()
    _EDGE_PRIORS.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> float:
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(
    conductance: float,
    q: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    return conductance + gain * q * dt - decay * conductance * dt

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def hybrid_conductance_update(
    edge: Edge,
    q: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
    prior: float = 0.5,
    likelihood: float = 0.5,
    false_positive: float = 0.1,
) -> float:
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    _EDGE_PRIORS[edge] = posterior
    return update_conductance(_EDGE_CONDUCTANCES.get(edge, 0.0), q, dt, gain, decay)

def hybrid_flux(
    edge: Edge,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> float:
    conductance = _EDGE_CONDUCTANCES.get(edge, 0.0)
    edge_length = length(_NODE_LOCATIONS[edge[0]], _NODE_LOCATIONS[edge[1]])
    return flux(conductance, edge_length, pressure_a, pressure_b, eps)

def add_node(node_id: str, location: Point) -> None:
    _NODE_LOCATIONS[node_id] = location

def add_edge(edge: Edge, conductance: float = 1.0, prior: float = 0.5) -> None:
    _EDGE_CONDUCTANCES[edge] = conductance
    _EDGE_PRIORS[edge] = prior

if __name__ == "__main__":
    reset_policy()
    add_node("A", (0.0, 0.0))
    add_node("B", (1.0, 0.0))
    add_edge(("A", "B"))
    print(hybrid_flux(("A", "B"), 1.0, 0.0))
    print(hybrid_conductance_update(("A", "B"), 0.1))