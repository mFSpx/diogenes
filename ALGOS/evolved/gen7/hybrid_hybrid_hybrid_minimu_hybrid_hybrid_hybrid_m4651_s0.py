# DARWIN HAMMER — match 4651, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2535_s1.py (gen6)
# born: 2026-05-29T23:57:12Z

"""
Module combining the hybrid minimum-cost tree Bayes update (hybrid_minimum_cost_tree_bayes_update_m6_s2.py) 
and the hybrid bandit-router and sketch-RLCT algorithm (hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py) 
with the entropy-based NLMS weight update from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1160_s0.py.

The mathematical bridge between the two parents is the use of expected values under probabilistic weights. 
The hybrid minimum-cost tree Bayes update uses expected edge lengths under posterior edge beliefs, 
while the hybrid bandit-router and sketch-RLCT algorithm uses expected rewards under log-count statistics. 
By fusing these two concepts, we obtain a novel algorithm that combines the strengths of both parents.

The hybrid algorithm replaces the deterministic edge contribution in the minimum-cost tree scoring 
with its expected value under the posterior edge belief. Similarly, the node distances are weighted 
by a node belief derived from incident edge posteriors. The bandit-router algorithm's log-count statistics 
are used to estimate the expected rewards of each action, which are then used to compute the posterior edge beliefs.

The entropy-based NLMS weight update from Parent A is used to scale the gradient learning of Parent B, 
resulting in a novel weight update that fuses the probabilistic edge weighting of Parent A with the 
gradient-descent learning of Parent B.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

Point = Tuple[float, float]
Edge = Tuple[str, str]
BanditAction = tuple[str, float, float, float, str]
class BanditAction:
    def __init__(self, action_id: str, propensity: float, expected_reward: float, confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    global _POLICY
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: list[BanditUpdate]) -> None:
    for update in updates:
        _POLICY[update.action_id] = [_POLICY.get(update.action_id, [0.0, 0.0])[0] + update.reward * update.propensity, _POLICY.get(update.action_id, [0.0, 0.0])[1] + update.propensity]

def compute_shannon_entropy(evidence: list[str]) -> float:
    """Return Shannon entropy H of a list of categorical tokens."""
    if not evidence:
        return 0.0
    counter = Counter(evidence)
    total = len(evidence)
    entropy = 0.0
    for cnt in counter.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy

def compute_edge_priors(edges: list[tuple[int, int]], evidence: list[str]) -> dict[tuple[int, int], float]:
    """
    Compute a prior probability πₑ for each edge using the same entropy H.
    All edges receive the same weight exp(-H) and are normalized.
    """
    H = compute_shannon_entropy(evidence)
    if not edges:
        return {}
    weight = math.exp(-H)
    total_weight = weight * len(edges)
    prior = weight / total_weight
    return {e: prior for e in edges}

def compute_posterior_edge_beliefs(edges: list[tuple[int, int]], edge_priors: dict[tuple[int, int], float], log_count_statistics: dict[str, float]) -> dict[tuple[int, int], float]:
    """
    Compute the posterior edge beliefs using the edge priors and log-count statistics.
    """
    posterior_beliefs = {}
    for edge in edges:
        posterior_beliefs[edge] = edge_priors[edge] * math.exp(log_count_statistics[edge])
    return posterior_beliefs

def hybrid_tree_cost(tree: dict[str, list[tuple[int, int]]], edge_priors: dict[tuple[int, int], float], log_count_statistics: dict[str, float], posterior_edge_beliefs: dict[tuple[int, int], float]) -> float:
    """
    Evaluate the hybrid cost using the posteriors and log-count statistics.
    """
    cost = 0.0
    for node in tree:
        for edge in tree[node]:
            cost += posterior_edge_beliefs[edge] * log_count_statistics[edge]
    return cost

def hybrid_bandit_router(tree: dict[str, list[tuple[int, int]]], actions: list[BanditAction], log_count_statistics: dict[str, float], posterior_edge_beliefs: dict[tuple[int, int], float]) -> float:
    """
    Run the hybrid bandit-router algorithm to estimate the expected rewards of each action.
    """
    expected_rewards = {}
    for action in actions:
        expected_reward = 0.0
        for edge in tree[action.action_id]:
            expected_reward += posterior_edge_beliefs[edge] * log_count_statistics[edge]
        expected_rewards[action.action_id] = expected_reward
    return expected_rewards

def hybrid_nlms_weight_update(weights: list[float], evidence: list[str], errors: list[float], x: list[float], epsilon: float) -> list[float]:
    """
    Update the weights using the NLMS algorithm with entropy-based scaling.
    """
    H = compute_shannon_entropy(evidence)
    learning_rate = math.exp(-H)
    weight_update = []
    for i in range(len(weights)):
        error = errors[i]
        x_i = x[i]
        weight_update_i = weights[i] + learning_rate * error * x_i / (np.linalg.norm(x_i)**2 + epsilon)
        weight_update.append(weight_update_i)
    return weight_update

if __name__ == "__main__":
    tree = {
        'A': [('A', 'B'), ('A', 'C')],
        'B': [('B', 'D')],
        'C': [('C', 'D')],
        'D': []
    }
    actions = [
        BanditAction('A', 0.5, 1.0, 0.1, 'B'),
        BanditAction('B', 0.3, 0.8, 0.2, 'C'),
        BanditAction('C', 0.7, 0.9, 0.1, 'D')
    ]
    log_count_statistics = {
        ('A', 'B'): 0.5,
        ('A', 'C'): 0.3,
        ('B', 'D'): 0.8,
        ('C', 'D'): 0.9
    }
    edge_priors = compute_edge_priors([(1, 2), (2, 3), (3, 4)], ['A', 'B', 'C', 'D'])
    posterior_edge_beliefs = compute_posterior_edge_beliefs([(1, 2), (2, 3), (3, 4)], edge_priors, log_count_statistics)
    hybrid_cost = hybrid_tree_cost(tree, edge_priors, log_count_statistics, posterior_edge_beliefs)
    expected_rewards = hybrid_bandit_router(tree, actions, log_count_statistics, posterior_edge_beliefs)
    weights = [1.0, 2.0, 3.0, 4.0]
    evidence = ['A', 'B', 'C', 'D']
    errors = [0.1, 0.2, 0.3, 0.4]
    x = [1.0, 2.0, 3.0, 4.0]
    epsilon = 0.1
    updated_weights = hybrid_nlms_weight_update(weights, evidence, errors, x, epsilon)
    print(hybrid_cost)
    print(expected_rewards)
    print(updated_weights)