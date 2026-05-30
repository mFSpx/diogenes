# DARWIN HAMMER — match 2650, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_koopman_operator_m64_s2.py (gen2)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_krampu_m293_s3.py (gen3)
# born: 2026-05-29T23:43:15Z

"""
Module for the Hybrid Algorithm, integrating the core topologies of 
hybrid_hybrid_bandit_router_koopman_operator_m64_s2.py and 
hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_krampu_m293_s3.py.
The mathematical bridge between the two structures is the application of the 
normalized least-mean-squares adaptive filter to the Koopman operator, 
enabling the analysis of the curvature of the connections between the different 
dimensions of the state space.
"""

import numpy as np
import random
import math
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

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

def koopman_operator(μ: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Apply the Koopman operator to the mean reward vector.

    Parameters
    ----------
    μ : np.ndarray
        Mean reward vector.
    K : np.ndarray
        Koopman operator matrix.

    Returns
    -------
    μ_next : np.ndarray
        Next mean reward vector.
    """
    μ_next = K @ μ
    return μ_next

def hybrid_select_action(μ: np.ndarray, K: np.ndarray, weights: np.ndarray, x: np.ndarray, store: float) -> BanditAction:
    """
    Select an action using the hybrid algorithm.

    Parameters
    ----------
    μ : np.ndarray
        Mean reward vector.
    K : np.ndarray
        Koopman operator matrix.
    weights : np.ndarray
        NLMS weights.
    x : np.ndarray
        Input feature vector.
    store : float
        Store value.

    Returns
    -------
    action : BanditAction
        Selected action.
    """
    μ_next = koopman_operator(μ, K)
    confidence_bound = (1 + store / (store + 1)) / np.sqrt(1 + np.linalg.norm(x))
    expected_reward = nlms_predict(weights, x) + confidence_bound * np.linalg.norm(μ_next)
    action_id = "hybrid_action"
    propensity = 1.0
    algorithm = "hybrid_hybrid"
    return BanditAction(action_id, propensity, expected_reward, confidence_bound, algorithm)

def hybrid_step(μ: np.ndarray, K: np.ndarray, weights: np.ndarray, x: np.ndarray, store: float, reward: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Update the hybrid algorithm state.

    Parameters
    ----------
    μ : np.ndarray
        Mean reward vector.
    K : np.ndarray
        Koopman operator matrix.
    weights : np.ndarray
        NLMS weights.
    x : np.ndarray
        Input feature vector.
    store : float
        Store value.
    reward : float
        Observed reward.

    Returns
    -------
    μ_next : np.ndarray
        Next mean reward vector.
    weights_next : np.ndarray
        Next NLMS weights.
    store_next : float
        Next store value.
    """
    μ_next = koopman_operator(μ, K)
    weights_next, _ = nlms_update(weights, x, reward)
    store_next = store + reward
    return μ_next, weights_next, store_next

def forecast_rewards(μ: np.ndarray, K: np.ndarray, h: int) -> np.ndarray:
    """
    Forecast future rewards using the Koopman operator.

    Parameters
    ----------
    μ : np.ndarray
        Mean reward vector.
    K : np.ndarray
        Koopman operator matrix.
    h : int
        Forecast horizon.

    Returns
    -------
    μ_forecast : np.ndarray
        Forecasted mean reward vector.
    """
    μ_forecast = np.linalg.matrix_power(K, h) @ μ
    return μ_forecast

if __name__ == "__main__":
    # Smoke test
    μ = np.array([1.0, 2.0])
    K = np.array([[0.5, 0.3], [0.2, 0.7]])
    weights = np.array([0.1, 0.2])
    x = np.array([0.3, 0.4])
    store = 1.0
    reward = 1.0
    action = hybrid_select_action(μ, K, weights, x, store)
    μ_next, weights_next, store_next = hybrid_step(μ, K, weights, x, store, reward)
    μ_forecast = forecast_rewards(μ, K, 2)
    print(action)
    print(μ_next)
    print(weights_next)
    print(store_next)
    print(μ_forecast)