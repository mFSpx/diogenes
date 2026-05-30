# DARWIN HAMMER — match 4217, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_bandit_m272_s2.py (gen4)
# born: 2026-05-29T23:54:09Z

"""
Hybrid Rectified Flow and NLMS-Bandit Engine
=============================================

This module fuses the Rectified Flow Matching algorithm (rectified_flow.py) 
with the Normalized Least-Mean-Squares (NLMS) adaptive filter and bandit 
router (hybrid_hybrid_hybrid_ternar_hybrid_hybrid_bandit_m272_s2.py). 
The mathematical bridge between the two structures is found by using the 
Rectified Flow's straight-line interpolant to generate input features for 
the NLMS predictor, which attempts to model the expected reward of the 
bandit router.

The Rectified Flow's straight-line interpolant is used to generate input 
vectors for the NLMS predictor, which are then used to predict the 
expected reward of the bandit router. The NLMS error is then used to 
adapt the global weight vector.

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
# Bandit utilities
# ----------------------------------------------------------------------

_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None: 
    _POLICY.clear()

def update_policy(updates: list) -> None:
    for u in updates:
        s=_POLICY.setdefault(u['action_id'],[0.0,0.0]); s[0]+=float(u['reward']); s[1]+=1.0

def _reward(a: str) -> float:
    total,n=_POLICY.get(a,[0.0,0.0]); return total/n if n else 0.0

def select_action(context: dict[str,float], actions: list[str], algorithm: str='linucb', epsilon: float=0.1, seed: int|str|None=7) -> dict:
    if not actions: raise ValueError('actions required')
    rng=random.Random(seed)
    if algorithm=='epsilon_greedy' and rng.random()<epsilon: chosen=rng.choice(actions)
    elif algorithm=='thompson': chosen=max(actions, key=lambda a: rng.betavariate(1+max(0,_reward(a)),1+max(0,1-_reward(a))))
    else:
        scale=math.sqrt(sum(float(v)*float(v) for v in context.values())) if context else 1.0
        chosen=max(actions, key=lambda a: _reward(a)+0.1*scale/math.sqrt(1+_POLICY.get(a,[0,0])[1]))
    return {'action_id': chosen, 'propensity': 1.0/len(actions), 'expected_reward': _reward(chosen), 'confidence_bound': 1.0/math.sqrt(1+_POLICY.get(chosen,[0,0])[1]), 'algorithm': algorithm}

def bayesian_update(prior: np.ndarray, likelihood: float, alpha: float, evidence: np.ndarray) -> np.ndarray:
    return (prior * likelihood * evidence) / (prior * likelihood * evidence + (1 - prior) * alpha)

def graph_update(graph: np.ndarray, action: dict) -> np.ndarray:
    graph[action['action_id']][action['action_id']] += 0.1
    return graph

def hybrid_operation(graph: np.ndarray, actions: list[str], context: dict[str,float], algorithm: str='linucb', epsilon: float=0.1, seed: int|str|None=7) -> dict:
    prior = np.ones(graph.shape[0]) / graph.shape[0]
    likelihood = 0.5
    alpha = 0.1
    evidence = np.random.rand(graph.shape[0])
    posterior = bayesian_update(prior, likelihood, alpha, evidence)

    action = select_action(context, actions, algorithm, epsilon, seed)
    update_policy([{'action_id': action['action_id'], 'reward': 1.0}])

    graph = graph_update(graph, action)

    return {'action': action, 'graph': graph, 'posterior': posterior}

def rectified_flow_nlms_bandit(weights: np.ndarray, graph: np.ndarray, actions: list[str], context: dict[str,float], 
                                x0: np.ndarray, x1: np.ndarray, t: float) -> Tuple[np.ndarray, dict]:
    interpolant_vector = interpolant(x0, x1, t)
    predicted_reward = nlms_predict(weights, interpolant_vector)
    action = select_action(context, actions)
    actual_reward = _reward(action['action_id'])
    weights, e = nlms_update(weights, interpolant_vector, actual_reward)
    return weights, hybrid_operation(graph, actions, context, algorithm='linucb', epsilon=0.1, seed=7)

def smoke_test():
    weights = np.random.rand(10)
    graph = np.random.rand(10, 10)
    actions = ['action1', 'action2', 'action3']
    context = {'feature1': 1.0, 'feature2': 2.0}
    x0 = np.random.rand(10)
    x1 = np.random.rand(10)
    t = 0.5
    weights, result = rectified_flow_nlms_bandit(weights, graph, actions, context, x0, x1, t)
    print("Weights:", weights)
    print("Result:", result)

if __name__ == "__main__":
    smoke_test()