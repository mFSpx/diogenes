# DARWIN HAMMER — match 2736, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:44:00Z

"""
Hybrid Algorithm: Fusing Physarum-Bandit-TTT Model and Hybrid Ternary Router

This module combines the Physarum-Bandit-TTT Model (hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py) 
and the Hybrid Ternary Router (hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py). 
The mathematical bridge between the two structures is the use of Bayesian update rules 
to update the priors of the edge costs in the ternary router, which are then used 
to compute the expected reward in the bandit model.

The governing equations of both parents are fused by introducing uncertainty 
in the edge costs of the tree, represented by prior probabilities. 
The Bayesian update rule is used to update these priors based on new evidence, 
which is then used to compute the expected cost of the tree and the reward signal 
that drives the bandit's learning update.
"""

import math
import random
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

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

@dataclass
class Edge:
    node_a: str
    node_b: str
    cost: float
    prior: float

_POLICY: dict[str, list[float]] = {}
_STORE: dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

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

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def hybrid_tree_cost(edges: List[Edge], path_weight: float = 0.2) -> float:
    material = 0.0
    for edge in edges:
        material += edge.cost * edge.prior
    return material + path_weight * sum(edge.cost for edge in edges)

def update_edge_priors(edges: List[Edge], updates: list[BanditUpdate]) -> List[Edge]:
    for edge in edges:
        for update in updates:
            if update.action_id == f"{edge.node_a}-{edge.node_b}":
                edge.prior = bayes_update(edge.prior, update.reward, bayes_marginal(edge.prior, update.reward, 0.1))
    return edges

def compute_expected_reward(edges: List[Edge]) -> float:
    expected_reward = 0.0
    for edge in edges:
        expected_reward += edge.cost * edge.prior
    return expected_reward

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(d_in, d_out))

if __name__ == "__main__":
    edges = [
        Edge("A", "B", 1.0, 0.5),
        Edge("B", "C", 2.0, 0.6),
        Edge("C", "A", 3.0, 0.7),
    ]

    updates = [
        BanditUpdate("context1", "A-B", 1.0, 0.5),
        BanditUpdate("context2", "B-C", 2.0, 0.6),
    ]

    updated_edges = update_edge_priors(edges, updates)
    expected_reward = compute_expected_reward(updated_edges)
    print(expected_reward)