# DARWIN HAMMER — match 4470, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_nlms_o_m2650_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_tropic_m2599_s0.py (gen6)
# born: 2026-05-29T23:55:54Z

"""
Module for the Hybrid Algorithm, integrating the core topologies of 
hybrid_hybrid_hybrid_bandit_hybrid_hybrid_nlms_o_m2650_s1.py and 
hybrid_hybrid_hybrid_distri_hybrid_hybrid_tropic_m2599_s0.py.
The mathematical bridge between the two structures is the application of the 
tropical max-plus algebra to the normalized least-mean-squares adaptive filter, 
enabling the analysis of the curvature of the connections between the different 
dimensions of the state space. The governing equations of the tropical max-plus 
algebra are used to compute the maximum expected utility of the decision hygiene 
scoring system, while the semantic weighting is used to compute the weighted edge 
costs in the graph-based leader election algorithm. The NLMS filter is used to update 
the weights of the actions in the bandit problem, and the tropical max-plus algebra 
is used to compute the maximum expected utility of the actions.
"""

import numpy as np
import random
import math
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections.abc import Mapping, Hashable

Node = Hashable
Graph = Mapping[Node, set[Node]]

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

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
    weights_update = weights + mu * e * x / (np.linalg.norm(x)**2 + eps)
    return weights_update, e

def t_add(x, y):
    """Tropical addition: max(x, y). Broadcasts."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication: x + y. Broadcasts."""
    return np.add(x, y)

def t_matmul(A, B):
    """Tropical matrix multiply.

    C[i, j] = max_k( A[i, k] + B[k, j] )

    A: (m, p), B: (p, n) → C: (m, n).
    Handles -inf entries correctly via numpy broadcasting.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcast: A[i, k, newaxis] + B[newaxis, k, j] then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def hybrid_nlms_tropical_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update using tropical max-plus algebra.

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
    weights_update = weights + mu * e * x / (np.linalg.norm(x)**2 + eps)
    weights_update = t_add(weights_update, np.zeros_like(weights_update))
    return weights_update, e

def compute_action_utility(
    action: BanditAction,
    weights: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Compute the utility of an action using the NLMS filter and tropical max-plus algebra.

    Parameters
    ----------
    action : BanditAction
        The action to compute the utility for.
    weights : np.ndarray
        The weights of the NLMS filter.
    x : np.ndarray
        The input feature vector.

    Returns
    -------
    utility : float
        The utility of the action.
    """
    prediction = nlms_predict(weights, x)
    utility = t_add(prediction, action.propensity)
    return utility

def hybrid_select_action(
    actions: List[BanditAction],
    weights: np.ndarray,
    x: np.ndarray,
) -> BanditAction:
    """
    Select an action using the hybrid algorithm.

    Parameters
    ----------
    actions : List[BanditAction]
        The list of available actions.
    weights : np.ndarray
        The weights of the NLMS filter.
    x : np.ndarray
        The input feature vector.

    Returns
    -------
    action : BanditAction
        The selected action.
    """
    utilities = [compute_action_utility(action, weights, x) for action in actions]
    max_utility = np.max(utilities)
    max_utility_index = np.argmax(utilities)
    return actions[max_utility_index]

if __name__ == "__main__":
    # Smoke test
    weights = np.array([0.5, 0.5])
    x = np.array([1.0, 1.0])
    target = 1.0
    weights_update, e = hybrid_nlms_tropical_update(weights, x, target)
    print("Weights update:", weights_update)
    print("Error:", e)

    actions = [
        BanditAction("action1", 0.5, 1.0, 0.1, "algorithm1"),
        BanditAction("action2", 0.5, 1.0, 0.1, "algorithm1"),
    ]
    selected_action = hybrid_select_action(actions, weights, x)
    print("Selected action:", selected_action.action_id)