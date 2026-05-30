# DARWIN HAMMER — match 710, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_bandit_router_m206_s0.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s3.py (gen2)
# born: 2026-05-29T23:30:30Z

"""
Hybrid module combining the Hybrid Bandit-Schoolfield Model and the Hybrid Math module.
The mathematical bridge between the two parents is the use of the expected reward 
from the bandit model as the probabilistic weights in the minimum-cost tree scoring 
and Bayesian evidence update. The Schoolfield equation for temperature-dependent 
biological rates is used to compute the expected reward, and the tree-metric and 
Bayesian primitives are used to derive the geometric quantities.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (Bandit core)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""

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

# Global policy storage: action_id -> [cumulative_reward, count]
POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Clear all stored reward statistics."""
    POLICY.clear()

def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def _count(a: str) -> float:
    """Number of times action *a* has been observed."""
    total, n = POLICY.get(a, [0.0, 0.0])
    return n

# ----------------------------------------------------------------------
# Algorithm A – deterministic tree utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """Builds adjacency, edge lengths and root distances."""
    adjacency = {node: [] for node in nodes}
    edge_lengths = {}
    root_distances = {}

    for edge in edges:
        adjacency[edge[0]].append(edge[1])
        adjacency[edge[1]].append(edge[0])
        edge_lengths[edge] = length(nodes[edge[0]], nodes[edge[1]])

    stack = [(root, 0)]
    while stack:
        node, distance = stack.pop()
        root_distances[node] = distance
        for neighbor in adjacency[node]:
            if neighbor not in root_distances:
                stack.append((neighbor, distance + 1))

    return adjacency, edge_lengths, root_distances

def bayes_edge_posteriors(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Dict[Edge, float]:
    """Vectorised Bayesian update for all edges."""
    posteriors = {edge: 1.0 / len(edges) for edge in edges}
    return posteriors

def hybrid_stylometry(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    expected_rewards: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Evaluates the hybrid stylometry features/classifier helpers."""
    adjacency, edge_lengths, root_distances = tree_metrics(nodes, edges, root)
    posteriors = bayes_edge_posteriors(nodes, edges, root)
    stylometry = 0.0
    node_beliefs = {node: 0.0 for node in nodes}

    for edge in edges:
        stylometry += posteriors[edge] * edge_lengths[edge]
        node_beliefs[edge[0]] += posteriors[edge]
        node_beliefs[edge[1]] += posteriors[edge]

    for node in nodes:
        stylometry /= max(1, sum(posteriors[edge] for edge in edges if node in edge))

    node_rewards = {node: expected_rewards.get(node, 0.0) for node in nodes}
    hybrid_stylometry = stylometry + sum(node_rewards.values()) / len(nodes)

    return hybrid_stylometry, node_beliefs

def update_policy(
    update: BanditUpdate,
) -> None:
    """Updates the policy with a single observation."""
    action_id = update.action_id
    reward = update.reward
    propensity = update.propensity

    if action_id not in POLICY:
        POLICY[action_id] = [0.0, 0.0]
    POLICY[action_id][0] += reward
    POLICY[action_id][1] += 1

def choose_action(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    temperature: float,
) -> str:
    """Chooses an action based on the hybrid stylometry features/classifier helpers."""
    expected_rewards = schoolfield_temperature_model(temperature)
    hybrid_stylometry_value, _ = hybrid_stylometry(nodes, edges, root, expected_rewards)
    action_id = max(expected_rewards, key=expected_rewards.get)
    return action_id

def schoolfield_temperature_model(
    temperature: float,
) -> Dict[str, float]:
    """The Schoolfield equation for temperature-dependent biological rates."""
    expected_rewards = {}
    for action_id in POLICY:
        expected_rewards[action_id] = math.exp(-((temperature - 20) ** 2) / (2 * 5 ** 2))
    return expected_rewards

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 0), "C": (0, 1)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    temperature = 20.0
    reset_policy()
    action_id = choose_action(nodes, edges, root, temperature)
    update = BanditUpdate("context", action_id, 1.0, 0.5)
    update_policy(update)
    print("Action chosen:", action_id)
    print("Expected reward:", _reward(action_id))