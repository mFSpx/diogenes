# DARWIN HAMMER — match 2736, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:44:00Z

"""
Hybrid algorithm combining the Physarum-Bandit-TTT Model from hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py 
and the Hybrid Ternary Router from hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py.
The mathematical bridge between the two structures is the notion of uncertainty 
in the tree edges and nodes, which can be updated using the Bayesian update rule 
and integrated into the routing decisions in the Physarum-Bandit-TTT Model.
The governing equations of both parents are fused by introducing uncertainty 
in the edge costs of the tree, represented by prior probabilities. 
The Bayesian update rule is used to update these priors based on new evidence, 
which is then used to compute the expected cost of the tree.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
_edge_priors: dict[tuple[str, str], float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    global _edge_priors
    _edge_priors.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0
    global _edge_priors
    for u in updates:
        edge = (u.context_id, u.action_id)
        prior = _edge_priors.get(edge, 0.5)
        likelihood = float(u.reward) / (float(u.reward) + u.confidence_bound)
        false_positive = 0.1
        marginal = bayes_marginal(prior, likelihood, false_positive)
        updated_prior = bayes_update(prior, likelihood, marginal)
        _edge_priors[edge] = updated_prior

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

def tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def hybrid_tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # Calculate conductance and flux for each edge
    conductances = {}
    fluxes = {}
    for a, b in edges:
        conductance = flux(conductance=1.0, edge_length=length(nodes[a], nodes[b]), pressure_a=1.0, pressure_b=1.0)
        fluxes[(a, b)] = flux(conductance=conductance, edge_length=length(nodes[a], nodes[b]), pressure_a=1.0, pressure_b=1.0)
        conductances[(a, b)] = conductance
    # Update edge priors based on flux
    for a, b in edges:
        _edge_priors[(a, b)] = bayes_update(_edge_priors.get((a, b), 0.5), fluxes[(a, b)], 0.1)
    # Calculate Bayesian evidence update for each edge
    edge_evidence = {}
    for a, b in edges:
        prior = _edge_priors[(a, b)]
        likelihood = fluxes[(a, b)]
        marginal = bayes_marginal(prior, likelihood, 0.1)
        edge_evidence[(a, b)] = bayes_update(prior, likelihood, marginal)
    # Calculate final tree cost
    material = 0.0
    for a, b in edges:
        material += length(nodes[a], nodes[b]) * edge_evidence[(a, b)]
    return material + path_weight * tree_cost(nodes, edges, root)

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((d_in, d_out)) * scale

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"
    path_weight = 0.2
    hybrid_tree_cost(nodes, edges, root, path_weight)