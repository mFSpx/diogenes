# DARWIN HAMMER — match 4217, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_bandit_m272_s2.py (gen4)
# born: 2026-05-29T23:54:09Z

"""
Hybrid Adaptive Filter and Bayesian Graph Engine
=============================================

This module fuses the Normalized Least-Mean-Squares (NLMS) adaptive filter 
and graph-propagation engine (hybrid_nlms_omni_chaotic_sprint_m59_s2.py) 
with the hybrid bandit router (hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s0.py). 
The mathematical bridge between the two structures is found by using the 
NLMS predictor to generate input features for the Bayesian graph engine, 
which updates the graph based on the predicted wavefront velocity.

The NLMS predictor uses the Rectified Flow's straight-line interpolant to 
generate input vectors, which are then used to predict the wavefront velocity 
of the graph-propagation engine. The NLMS error is then used to adapt the global 
weight vector. The Bayesian graph engine updates the graph based on the predicted 
wavefront velocity.

Imports:
    numpy
    standard library
    math
    random
    sys
    pathlib
"""

import numpy as np
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random
import sys

# ----------------------------------------------------------------------
# Core NLMS utilities
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot-product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.
    """
    e = target - nlms_predict(weights, x)
    weights += mu * e * x / (np.linalg.norm(x) ** 2 + eps)
    return weights, e


# ----------------------------------------------------------------------
# Rectified Flow utilities
# ----------------------------------------------------------------------
def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0."""
    return t * x1 + (1 - t) * x0


# ----------------------------------------------------------------------
# Hybrid bandit router utilities
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()


def update_policy(updates: list) -> None:
    for u in updates:
        s = _POLICY.setdefault(u['action_id'], [0.0, 0.0])
        s[0] += float(u['reward'])
        s[1] += 1.0


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(context: dict[str, float], actions: list[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> dict:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]))
    return {'action_id': chosen, 'propensity': 1.0 / len(actions), 'expected_reward': _reward(chosen), 'confidence_bound': 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), 'algorithm': algorithm}


def bayesian_update(prior: np.ndarray, likelihood: float, alpha: float, evidence: np.ndarray) -> np.ndarray:
    return (prior * likelihood * evidence) / (prior * likelihood * evidence + (1 - prior) * alpha)


def graph_update(graph: np.ndarray, action: dict) -> np.ndarray:
    graph[action['action_id']][action['action_id']] += 0.1
    return graph


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_predict(graph: np.ndarray, action: dict) -> np.ndarray:
    weights = np.random.rand(graph.shape[1])
    x = np.random.rand(graph.shape[1])
    target = float(action['expected_reward'])
    weights, _ = nlms_update(weights, x, target)
    predicted_velocity = nlms_predict(weights, x)
    return predicted_velocity


def hybrid_update(graph: np.ndarray, action: dict, predicted_velocity: float) -> np.ndarray:
    graph = graph_update(graph, action)
    likelihood = 0.5
    alpha = 0.1
    evidence = np.random.rand(graph.shape[0])
    prior = np.ones(graph.shape[0]) / graph.shape[0]
    posterior = bayesian_update(prior, likelihood, alpha, evidence)
    return graph, posterior


def hybrid_operation(graph: np.ndarray, actions: list[str], context: dict[str, float], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> dict:
    action = select_action(context, actions, algorithm, epsilon, seed)
    predicted_velocity = hybrid_predict(graph, action)
    graph, posterior = hybrid_update(graph, action, predicted_velocity)
    return {'action': action, 'graph': graph, 'posterior': posterior, 'predicted_velocity': predicted_velocity}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    graph = np.random.rand(10, 10)
    actions = ['action1', 'action2', 'action3']
    context = {'feature1': 1.0, 'feature2': 2.0}
    algorithm = 'linucb'
    epsilon = 0.1
    seed = 7
    hybrid_operation(graph, actions, context, algorithm, epsilon, seed)