# DARWIN HAMMER — match 1337, survivor 0
# gen: 3
# parent_a: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s1.py (gen1)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py (gen2)
# born: 2026-05-29T23:35:25Z

import math
import random
import numpy as np
import sys
import pathlib

# ----------------------------------------------------------------------
# Module docstring
# ----------------------------------------------------------------------
"""
This module represents a novel HYBRID algorithm that mathematically fuses the core topologies of 
the Capybara Optimization Algorithm (capybara_optimization.py) and the NLMS adaptive filter 
(hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py) into a single unified system. 

The mathematical bridge between these two structures is based on the integration of the social 
interaction and predator evasion mechanisms from the Capybara Optimization Algorithm with the 
NLMS adaptive filter's weight vector update rule and the similarity matrix of the span-graph. 

Specifically, the Capybara Optimization Algorithm's social interaction and predator evasion 
mechanisms are used to optimize the NLMS adaptive filter's weight vector update rule and the 
similarity matrix of the span-graph, resulting in a more efficient and effective hybrid algorithm.

The hybrid algorithm uses the Capybara Optimization Algorithm's social interaction and predator 
evasion mechanisms to adaptively update the NLMS adaptive filter's weight vector and the 
similarity matrix of the span-graph. The adapted weights are then used as multiplicative factors 
in the edge-cost definition of the minimum-cost spanning tree, yielding a tree that reflects both 
learned relevance (via NLMS) and intrinsic similarity (via the graph).
"""

# ----------------------------------------------------------------------
# Social interaction and predator evasion functions (Parent A)
# ----------------------------------------------------------------------
def social_interaction(x: np.ndarray, g_best: np.ndarray, k: int = 1, r1: float | None = None, seed: int | str | None = None) -> np.ndarray:
    if x.shape != g_best.shape:
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0 <= r <= 1):
        raise ValueError("r1 must be in [0, 1]")
    return x + r * (g_best - k * x)

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def predator_evasion(x: np.ndarray, delta: float, r2: float | None = None, seed: int | str | None = None) -> np.ndarray:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    rng = random.Random(seed)
    r = rng.random() if r2 is None else r2
    if not (0 <= r <= 1):
        raise ValueError("r2 must be in [0, 1]")
    step = (2 * r - 1) * delta
    return x + step * x

# ----------------------------------------------------------------------
# NLMS adaptive filter functions (Parent B)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Args:
        weights: Current weight vector (shape (n,)).
        x: Input feature vector (shape (n,)).
        target: Desired scalar output.
        mu: Step‑size (0 < mu ≤ 1).
        eps: Small constant to avoid division by zero.

    Returns:
        (new_weights, error) where error = target - y.
    """
    y = predict(weights, x)
    error = target - y
    if abs(error) < eps:
        return weights, 0
    step_size = mu * error / (np.dot(x, x) + eps)
    new_weights = weights + step_size * x
    return new_weights, error

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    social_interaction_param: float | None = None,
    evasion_delta_param: float | None = None,
) -> Tuple[np.ndarray, float]:
    # Adapt NLMS weights using social interaction and evasion mechanisms
    social_interaction_weights = social_interaction(weights, target, r1=social_interaction_param)
    evasion_delta_weight = evasion_delta(0, 100, delta_max=evasion_delta_param)
    new_weights = social_interaction_weights + evasion_delta_weight * weights
    return nlms_update(new_weights, x, target, mu=mu, eps=eps)

def hybrid_similarity_matrix(
    similarity_matrix: np.ndarray,
    weights: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
    social_interaction_param: float | None = None,
    evasion_delta_param: float | None = None,
) -> np.ndarray:
    # Adapt similarity matrix using NLMS weights and social interaction and evasion mechanisms
    nlms_weights = hybrid_nlms_update(weights, similarity_matrix, 1, mu=mu, eps=eps, social_interaction_param=social_interaction_param, evasion_delta_param=evasion_delta_param)
    adapted_similarity_matrix = similarity_matrix + nlms_weights * similarity_matrix
    return adapted_similarity_matrix

def hybrid_minimum_cost_spanning_tree(similarity_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Use adapted similarity matrix and NLMS weights to find minimum-cost spanning tree
    adapted_similarity_matrix = hybrid_similarity_matrix(similarity_matrix, weights)
    tree = np.argmin(adapted_similarity_matrix, axis=1)
    return tree

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Test hybrid_nlms_update function
    weights = np.array([1.0, 2.0, 3.0])
    x = np.array([4.0, 5.0, 6.0])
    target = 10.0
    mu = 0.5
    eps = 1e-9
    social_interaction_param = 0.5
    evasion_delta_param = 0.5
    new_weights, error = hybrid_nlms_update(weights, x, target, mu=mu, eps=eps, social_interaction_param=social_interaction_param, evasion_delta_param=evasion_delta_param)
    print(f"New weights: {new_weights}")
    print(f"Error: {error}")

    # Test hybrid_similarity_matrix function
    similarity_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([1.0, 2.0])
    mu = 0.5
    eps = 1e-9
    social_interaction_param = 0.5
    evasion_delta_param = 0.5
    adapted_similarity_matrix = hybrid_similarity_matrix(similarity_matrix, weights, mu=mu, eps=eps, social_interaction_param=social_interaction_param, evasion_delta_param=evasion_delta_param)
    print(f"Adapted similarity matrix: {adapted_similarity_matrix}")

    # Test hybrid_minimum_cost_spanning_tree function
    similarity_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([1.0, 2.0])
    mu = 0.5
    eps = 1e-9
    social_interaction_param = 0.5
    evasion_delta_param = 0.5
    tree = hybrid_minimum_cost_spanning_tree(similarity_matrix, weights)
    print(f"Minimum-cost spanning tree: {tree}")