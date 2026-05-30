# DARWIN HAMMER — match 994, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_state_space_duality_m143_s1.py (gen2)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s0.py (gen3)
# born: 2026-05-29T23:32:05Z

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random
import sys
from pathlib import Path
import math

@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters for the schoolfield model."""
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Erase all learned statistics."""
    global _POLICY
    _POLICY.clear()

def _reward(action: str) -> float:
    """Empirical mean reward for *action* (0 if never observed)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observations into the global policy."""
    global _POLICY
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: Dict[str, Point], edges: Dict[Tuple[str, str], float]) -> float:
    """Calculate the minimum cost of a tree."""
    total_cost = 0
    for edge in edges.values():
        total_cost += edge
    return total_cost

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Calculate the developmental rate based on the schoolfield model."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    denominator = 1.0 + low + high
    if denominator == 0:
        raise ValueError("denominator cannot be zero")
    return numerator / denominator

def temperature_dependent_state_transition(A: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    """Calculate the state transition matrix based on the schoolfield model."""
    rate = developmental_rate(temp_k, params)
    return A * rate

def hybrid_ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
    nodes: Dict[str, Point] = {},
    edges: Dict[Tuple[str, str], float] = {}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the next state based on the hybrid model."""
    state_transition_matrix = temperature_dependent_state_transition(A, temp_k, params)
    output_projection_matrix = temperature_dependent_output_projection(C, temp_k, params)
    tree_cost_value = tree_cost(nodes, edges)
    confidence_bound = _count(edges.keys()[0]) / (tree_cost_value + 1)
    expected_reward = _reward(edges.keys()[0])
    return state_transition_matrix @ h, x + B @ h, output_projection_matrix @ h + C @ x + expected_reward * confidence_bound

def temperature_dependent_output_projection(C: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    """Calculate the output projection matrix based on the schoolfield model."""
    rate = developmental_rate(temp_k, params)
    return C * rate

def hybrid_bandit_update(
    context_id: str,
    action_id: str,
    reward: float,
    propensity: float,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
    nodes: Dict[str, Point] = {},
    edges: Dict[Tuple[str, str], float] = {}
) -> None:
    """Update the policy based on the hybrid model."""
    global _POLICY
    stats = _POLICY.setdefault(action_id, [0.0, 0.0])
    stats[0] += float(reward)
    stats[1] += 1.0
    edges[(context_id, action_id)] = length(nodes.get(context_id, Point(0, 0)), nodes.get(action_id, Point(0, 0)))
    developmental_rate_value = developmental_rate(temp_k, params)
    confidence_bound = developmental_rate_value * (stats[1] / (stats[1] + 1))
    expected_reward = _reward(action_id)
    _POLICY.setdefault("confidence_bound", [0.0, 0.0])[0] += float(confidence_bound)
    _POLICY.setdefault("expected_reward", [0.0, 0.0])[0] += float(expected_reward)

if __name__ == "__main__":
    # Smoke test
    nodes = {"a": Point(1, 2), "b": Point(3, 4)}
    edges = {(("a", "b"), 0)}
    temp_k = 300
    h = np.array([1, 2])
    x = np.array([3, 4])
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    params = SchoolfieldParams()
    hybrid_ssm_step(h, x, A, B, C, temp_k, params, nodes, edges)
    hybrid_bandit_update("a", "b", 1.0, 0.5, temp_k, params, nodes, edges)