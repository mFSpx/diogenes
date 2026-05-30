# DARWIN HAMMER — match 994, survivor 0
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

"""
This module fuses the governing equations of two independent prototypes:
* **hybrid_hybrid_bandit_router_state_space_duality_m143_s1.py** — a hybrid bandit-store algorithm that combines a lightweight contextual bandit router with a state-space duality primitive.
* **hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s0.py** — a minimum-cost tree scoring algorithm for length/path trade-offs combined with a hybrid bandit algorithm.

The mathematical bridge is built on the observation that the tree structure can be used to modulate the confidence term of the bandit, creating a coupled system that integrates the governing equations of both parents. The tree's nodes are used to represent the bandit's context, and the edges are used to calculate the distance between contexts, which is then used to update the bandit's policy.

The hybrid operation is achieved by using the state-space duality to update the tree structure, and the minimum-cost tree scoring to update the bandit's confidence term.
"""

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

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

@dataclass(frozen=True)
class Point:
    x: float
    y: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def length(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
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
    rate = developmental_rate(temp_k, params)
    return A * rate

def temperature_dependent_output_projection(C: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return C * rate

def hybrid_ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> Tuple[np.ndarray, np.ndarray]:
    rate = developmental_rate(temp_k, params)
    x = np.dot(A * rate, x) + np.dot(B, h)
    y = np.dot(C * rate, x)
    return x, y

def update_bandit_policy_with_tree_context(nodes: Dict[str, Point], edge_matrix: np.ndarray, updates: List[BanditUpdate]) -> None:
    for u in updates:
        context_id = u.context_id
        node = nodes[context_id]
        nearest_nodes = []
        for n in nodes:
            if n != context_id:
                nearest_nodes.append((n, length(node, nodes[n])))
        nearest_nodes.sort(key=lambda x: x[1])
        confidence_bound = 0.0
        for n, dist in nearest_nodes[:5]:
            confidence_bound += 1.0 / (1.0 + dist)
        updates.append(BanditUpdate(context_id, u.action_id, u.reward, confidence_bound))
    update_policy(updates)

def update_tree_with_bandit_policy(nodes: Dict[str, Point], edge_matrix: np.ndarray, updates: List[BanditUpdate]) -> None:
    for u in updates:
        context_id = u.context_id
        node = nodes[context_id]
        new_node = Point(node.x + u.propensity * random.uniform(-1.0, 1.0), node.y + u.propensity * random.uniform(-1.0, 1.0))
        nodes[context_id] = new_node

if __name__ == "__main__":
    nodes = {"A": Point(0.0, 0.0), "B": Point(10.0, 0.0), "C": Point(5.0, 10.0)}
    edge_matrix = np.array([[0.0, 10.0, 5.0], [10.0, 0.0, 10.0], [5.0, 10.0, 0.0]])
    updates = [BanditUpdate("A", "action1", 1.0, 0.5), BanditUpdate("B", "action2", 2.0, 0.7)]
    update_bandit_policy_with_tree_context(nodes, edge_matrix, updates)
    update_tree_with_bandit_policy(nodes, edge_matrix, updates)
    print("Nodes:", nodes)
    print("Updates:", updates)