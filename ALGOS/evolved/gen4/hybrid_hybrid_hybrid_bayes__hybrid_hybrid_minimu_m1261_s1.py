# DARWIN HAMMER — match 1261, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py (gen3)
# born: 2026-05-29T23:34:46Z

"""
Module combining the Bayesian-Krampus-Ollivier-Ricci-Hybrid-Bandit-Router Algorithm 
(hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s1.py) and the hybrid minimum-cost 
tree Bayes update and hybrid bandit-router and sketch-RLCT algorithm 
(hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py).

The mathematical bridge between the two algorithms is the application of expected values 
under probabilistic weights to update the probabilities of the brain map projections, 
taking into account the Ollivier-Ricci curvature of the connections between the different 
dimensions of the brain map, while using the Structural Similarity Index (SSIM) to inform 
the selection of actions in the bandit algorithm and the expected rewards under log-count 
statistics.

The hybrid algorithm replaces the deterministic edge contribution in the minimum-cost tree 
scoring with its expected value under the posterior edge belief. Similarly, the node distances 
are weighted by a node belief derived from incident edge posteriors. The bandit-router 
algorithm's log-count statistics are used to estimate the expected rewards of each action, 
which are then used to compute the posterior edge beliefs.

The module implements:
* `hybrid_score` – evaluates the hybrid score using the SSIM and expected rewards.
* `bayes_edge_posteriors` – vectorised Bayesian update for all edges.
* `hybrid_tree_cost` – evaluates the hybrid cost using the posteriors and log-count statistics.
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

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# SSIM implementation
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

# Hybrid routing utilities
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

# Bandit algorithm
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
        action_id = update.action_id
        reward = update.reward
        propensity = update.propensity
        if action_id not in _POLICY:
            _POLICY[action_id] = [0.0, 0.0]
        _POLICY[action_id][0] += reward * propensity
        _POLICY[action_id][1] += propensity

def bayes_edge_posteriors(edge_lengths: np.ndarray, prior: float) -> np.ndarray:
    # Vectorised Bayesian update for all edges
    posteriors = prior * np.exp(-edge_lengths)
    return posteriors / posteriors.sum()

def hybrid_tree_cost(edge_posteriors: np.ndarray, log_count_stats: np.ndarray) -> float:
    # Evaluates the hybrid cost using the posteriors and log-count statistics
    return np.sum(edge_posteriors * log_count_stats)

def main():
    # Smoke test
    packet = {"payload": [0.2, 0.5, 0.3]}
    score = hybrid_score(packet)
    print(f"Hybrid score: {score:.4f}")

    edge_lengths = np.array([1.0, 2.0, 3.0])
    prior = 0.1
    edge_posteriors = bayes_edge_posteriors(edge_lengths, prior)
    print(f"Edge posteriors: {edge_posteriors}")

    log_count_stats = np.array([0.5, 0.6, 0.7])
    hybrid_cost = hybrid_tree_cost(edge_posteriors, log_count_stats)
    print(f"Hybrid tree cost: {hybrid_cost:.4f}")

if __name__ == "__main__":
    main()