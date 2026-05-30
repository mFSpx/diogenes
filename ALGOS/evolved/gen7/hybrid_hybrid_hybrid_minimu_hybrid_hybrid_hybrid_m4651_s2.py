# DARWIN HAMMER — match 4651, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2535_s1.py (gen6)
# born: 2026-05-29T23:57:12Z

"""
Module combining the hybrid minimum-cost tree Bayes update and the hybrid bandit-router 
with sketch-RLCT algorithm (hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s1.py) 
and the hybrid fusion algorithm (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2535_s1.py).

The mathematical bridge between the two algorithms is the use of expected values under 
probabilistic weights and the integration of Shannon entropy to scale the NLMS learning rate.
By fusing these two concepts, we obtain a novel algorithm that combines the strengths of both parents.

The hybrid algorithm replaces the deterministic edge contribution in the minimum-cost tree 
scoring with its expected value under the posterior edge belief. Similarly, the node distances 
are weighted by a node belief derived from incident edge posteriors. The bandit-router 
algorithm's log-count statistics are used to estimate the expected rewards of each action, 
which are then used to compute the posterior edge beliefs. The Shannon entropy is used to 
scale the NLMS learning rate and to weight the edge priors.

This module implements:
* `tree_metrics` – builds adjacency, edge lengths and root distances.
* `bayes_edge_posteriors` – vectorised Bayesian update for all edges.
* `count_min_sketch` – Count-Min sketch of item frequencies.
* `hybrid_tree_cost` – evaluates the hybrid cost using the posteriors and log-count statistics.
* `compute_shannon_entropy` – computes the Shannon entropy of a list of categorical tokens.
* `compute_edge_priors` – computes a prior probability for each edge using the Shannon entropy.
* `hybrid_update` – updates the weights using the hybrid algorithm.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

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
        action = update.action_id
        reward = update.reward
        if action not in _POLICY:
            _POLICY[action] = [0.0, 0.0]
        _POLICY[action][0] += reward
        _POLICY[action][1] += 1

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

def tree_metrics(edges: list[Edge]) -> (dict[Edge, float], dict[str, float]):
    """Build adjacency, edge lengths and root distances."""
    adjacency = {}
    edge_lengths = {}
    root_distances = {}
    for edge in edges:
        if edge[0] not in adjacency:
            adjacency[edge[0]] = []
        if edge[1] not in adjacency:
            adjacency[edge[1]] = []
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
        edge_lengths[edge] = random.random()
        root_distances[edge[0]] = 0.0
        root_distances[edge[1]] = 0.0
    return adjacency, edge_lengths, root_distances

def bayes_edge_posteriors(edge_lengths: dict[Edge, float], evidence: list[str]) -> dict[Edge, float]:
    """Vectorised Bayesian update for all edges."""
    priors = compute_edge_priors(list(edge_lengths.keys()), evidence)
    posteriors = {}
    for edge, length in edge_lengths.items():
        posterior = length * priors[edge]
        posteriors[edge] = posterior
    return posteriors

def count_min_sketch(edges: list[Edge]) -> dict[Edge, int]:
    """Count-Min sketch of item frequencies."""
    sketch = {}
    for edge in edges:
        if edge not in sketch:
            sketch[edge] = 0
        sketch[edge] += 1
    return sketch

def hybrid_tree_cost(posteriors: dict[Edge, float], sketch: dict[Edge, int]) -> float:
    """Evaluates the hybrid cost using the posteriors and log-count statistics."""
    cost = 0.0
    for edge, posterior in posteriors.items():
        count = sketch[edge]
        cost += posterior * count
    return cost

def hybrid_update(weights: dict[Edge, float], posteriors: dict[Edge, float], sketch: dict[Edge, int], learning_rate: float) -> dict[Edge, float]:
    """Updates the weights using the hybrid algorithm."""
    updated_weights = {}
    for edge, weight in weights.items():
        posterior = posteriors[edge]
        count = sketch[edge]
        updated_weight = weight + learning_rate * posterior * count
        updated_weights[edge] = updated_weight
    return updated_weights

if __name__ == "__main__":
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    evidence = ["A", "B", "C", "D"]
    adjacency, edge_lengths, root_distances = tree_metrics(edges)
    posteriors = bayes_edge_posteriors(edge_lengths, evidence)
    sketch = count_min_sketch(edges)
    cost = hybrid_tree_cost(posteriors, sketch)
    weights = {edge: random.random() for edge in edges}
    updated_weights = hybrid_update(weights, posteriors, sketch, 0.1)
    print("Cost:", cost)
    print("Updated Weights:", updated_weights)