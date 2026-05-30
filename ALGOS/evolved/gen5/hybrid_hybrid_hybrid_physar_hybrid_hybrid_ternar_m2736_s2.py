# DARWIN HAMMER — match 2736, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:44:00Z

"""
Hybrid Algorithm: Fusing Physarum-Bandit-TTT Model with Hybrid Ternary Router
================================================================================

This module fuses the Physarum-Bandit-TTT model (hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py) 
with the Hybrid Ternary Router (hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py). 
The mathematical bridge between the two structures is the identification of 
**uncertainty** in the tree edges and nodes of the Hybrid Ternary Router with 
**confidence bounds** of the bandit actions in the Physarum-Bandit-TTT model. 
The Bayesian update rule is used to update the priors of the tree edges, 
which are then used to compute the expected cost of the tree. 
The conductance update rule from the Physarum model is used to update 
the confidence bounds of the bandit actions.

Imports:
    numpy as np
    standard library: math, random, sys, pathlib
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass

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

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]], 
    edges: List[Tuple[str, str]], 
    root: str, 
    edge_priors: Dict[Tuple[str, str], float], 
    path_weight: float = 0.2
) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
                stack.append(b)
    expected_cost = material + path_weight * sum(dist.values())
    uncertainty = 0.0
    for edge in edges:
        uncertainty += bayes_marginal(edge_priors[edge], 0.5, 0.1)
    return expected_cost + uncertainty

def update_bandit_action(
    action: BanditAction, 
    reward: float, 
    edge_priors: Dict[Tuple[str, str], float]
) -> BanditAction:
    new_confidence_bound = action.confidence_bound + bayes_update(edge_priors.get((action.action_id, 'edge'), 0.5), 0.5, 0.6)
    return BanditAction(
        action.action_id, 
        action.propensity, 
        action.expected_reward + reward, 
        new_confidence_bound, 
        action.algorithm
    )

def hybrid_operation(
    nodes: Dict[str, Tuple[float, float]], 
    edges: List[Tuple[str, str]], 
    root: str, 
    edge_priors: Dict[Tuple[str, str], float], 
    bandit_actions: List[BanditAction]
) -> Tuple[float, List[BanditAction]]:
    expected_cost = hybrid_tree_cost(nodes, edges, root, edge_priors)
    updated_bandit_actions = []
    for action in bandit_actions:
        reward = _reward(action.action_id)
        updated_action = update_bandit_action(action, reward, edge_priors)
        updated_bandit_actions.append(updated_action)
    return expected_cost, updated_bandit_actions

if __name__ == "__main__":
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 0.0), 'C': (1.0, 1.0)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    edge_priors = {('A', 'B'): 0.5, ('B', 'C'): 0.6, ('C', 'A'): 0.7}
    bandit_actions = [
        BanditAction('action1', 0.5, 10.0, 0.1, 'algorithm1'),
        BanditAction('action2', 0.6, 20.0, 0.2, 'algorithm2')
    ]
    expected_cost, updated_bandit_actions = hybrid_operation(nodes, edges, root, edge_priors, bandit_actions)
    print(expected_cost)
    for action in updated_bandit_actions:
        print(action)