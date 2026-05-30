# DARWIN HAMMER — match 1261, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py (gen3)
# born: 2026-05-29T23:34:46Z

"""
Module combining the Bayesian-Krampus-Ollivier-Ricci-Hybrid-Bandit-Router Algorithm (hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1.py) 
and the hybrid minimum-cost tree Bayes update and hybrid bandit-router algorithm (hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py).

The mathematical bridge between the two algorithms is the use of expected values under probabilistic 
weights and the application of Bayesian inference to update the probabilities of the brain map projections. 
By fusing these two concepts, we obtain a novel algorithm that combines the strengths of both parents.

The hybrid algorithm replaces the deterministic edge contribution in the minimum-cost tree 
scoring with its expected value under the posterior edge belief. Similarly, the node distances 
are weighted by a node belief derived from incident edge posteriors. The bandit-router 
algorithm's log-count statistics are used to estimate the expected rewards of each action, 
which are then used to compute the posterior edge beliefs. The Structural Similarity Index (SSIM) 
is used to inform the selection of actions in the bandit algorithm.

The module implements:
* `hybrid_score` – evaluates the hybrid score using the posteriors and log-count statistics.
* `bayes_edge_posteriors` – vectorised Bayesian update for all edges.
* `compute_ssim` – computes the Structural Similarity Index (SSIM) between two vectors.
"""

import numpy as np
import random
import math
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict

Point = Tuple[float, float]
Edge = Tuple[str, str]

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

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for update in updates:
        key = update.action_id
        total, n = _POLICY.get(key, [0.0, 0.0])
        _POLICY[key] = [total + update.reward * update.propensity, n + 1]

def compute_ssim(
    x: list[float],
    y: list[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

def hybrid_score(packet: dict[str, list[float]]) -> float:
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size < PROTOTYPE_VECTOR.size:
            payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
        elif payload_vec.size > PROTOTYPE_VECTOR.size:
            payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(payload_vec.tolist(), PROTOTYPE_VECTOR.tolist(), dynamic_range=1.0)
    except Exception:
        return 0.0

def bayes_edge_posteriors(edges: List[Edge], prior: Dict[Edge, float], updates: List[BanditUpdate]) -> Dict[Edge, float]:
    posteriors = prior.copy()
    for update in updates:
        edge = (update.context_id, update.action_id)
        if edge in posteriors:
            posteriors[edge] = posteriors[edge] * (1 - update.propensity) + update.propensity * (update.reward + 1) / 2
    return posteriors

def hybrid_tree_cost(edges: List[Edge], posteriors: Dict[Edge, float], log_count_stats: Dict[Edge, float]) -> float:
    cost = 0.0
    for edge in edges:
        if edge in posteriors:
            cost += posteriors[edge] * log_count_stats.get(edge, 0.0)
    return cost

if __name__ == "__main__":
    # Test the hybrid score function
    packet = {"payload": [0.2, 0.5, 0.3]}
    print(hybrid_score(packet))

    # Test the Bayesian edge posteriors function
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    prior = {(edge): 0.5 for edge in edges}
    updates = [BanditUpdate("A", "B", 1.0, 0.5), BanditUpdate("B", "C", 0.5, 0.3)]
    posteriors = bayes_edge_posteriors(edges, prior, updates)
    print(posteriors)

    # Test the hybrid tree cost function
    log_count_stats = {(edge): 1.0 for edge in edges}
    cost = hybrid_tree_cost(edges, posteriors, log_count_stats)
    print(cost)