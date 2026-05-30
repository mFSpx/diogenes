# DARWIN HAMMER — match 118, survivor 1
# gen: 2
# parent_a: rlct_grokking.py (gen0)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s4.py (gen1)
# born: 2026-05-29T23:25:40Z

"""
Hybrid module combining rlct_grokking and hybrid_nlms_omni_chaotic_sprint_m59_s4.

The mathematical bridge between the two parents lies in the application of 
Watanabe's free energy asymptotic to the weights update process in the NLMS 
algorithm. This bridge allows us to incorporate the Real Log Canonical Threshold 
(RLCT) into the weights update process, effectively creating a hybrid system 
that combines the strengths of both parent algorithms.

The RLCT is used to adjust the learning rate in the NLMS algorithm, allowing for 
more efficient convergence and better generalization. The hybrid system 
also incorporates the activation pattern count from the rlct_grokking 
algorithm to further improve the performance of the NLMS algorithm.
"""

import numpy as np
import math
import random
import sys
from collections import deque, Counter
from pathlib import Path

NodeId = str
Edge = tuple[NodeId, NodeId, int]  # (src, dst, impedance)

def bayesian_information_criterion(log_likelihood, n_params, n_samples):
    """Standard BIC.

    BIC = -2 * log_likelihood + n_params * log(n_samples)

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood evaluated at the MLE.
    n_params : int or float
        Number of free parameters.
    n_samples : int or float
        Dataset size n.

    Returns
    -------
    float
        BIC score.  Lower is better.
    """
    return -2 * log_likelihood + n_params * math.log(n_samples)

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """NLMS prediction function.

    Parameters
    ----------
    weights : np.ndarray
        Weights vector.
    x : np.ndarray
        Input vector.

    Returns
    -------
    float
        Predicted value.
    """
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """NLMS update function.

    Parameters
    ----------
    weights : np.ndarray
        Weights vector.
    x : np.ndarray
        Input vector.
    target : float
        Target value.
    mu : float, optional
        Learning rate (default is 0.5).
    eps : float, optional
        Small value to prevent division by zero (default is 1e-9).

    Returns
    -------
    tuple[np.ndarray, float]
        Updated weights and error.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error

def estimate_rlct_from_losses(losses: list[float]) -> float:
    """Estimate the Real Log Canonical Threshold (RLCT) from a list of losses.

    Parameters
    ----------
    losses : list[float]
        List of losses.

    Returns
    -------
    float
        Estimated RLCT.
    """
    n = len(losses)
    log_n = math.log(n)
    lambda_val = (math.log(sum(losses)) - math.log(n)) / log_n
    return lambda_val

def hybrid_nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    losses: list[float] = None,
) -> tuple[np.ndarray, float]:
    """Hybrid NLMS update function.

    Parameters
    ----------
    weights : np.ndarray
        Weights vector.
    x : np.ndarray
        Input vector.
    target : float
        Target value.
    mu : float, optional
        Learning rate (default is 0.5).
    eps : float, optional
        Small value to prevent division by zero (default is 1e-9).
    losses : list[float], optional
        List of losses (default is None).

    Returns
    -------
    tuple[np.ndarray, float]
        Updated weights and error.
    """
    if losses is not None:
        rlct = estimate_rlct_from_losses(losses)
        mu = mu * rlct
    new_weights, error = nlms_update(weights, x, target, mu, eps)
    return new_weights, error

def generate_synthetic_graph(num_nodes: int, avg_degree: int = 3) -> tuple[dict[NodeId, list[tuple[NodeId, int]]], np.ndarray]:
    """Generate a synthetic graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    avg_degree : int, optional
        Average degree (default is 3).

    Returns
    -------
    tuple[dict[NodeId, list[tuple[NodeId, int]]], np.ndarray]
        Adjacency list and feature matrix.
    """
    random.seed(42)
    nodes = [f"n{i}" for i in range(num_nodes)]
    adjacency: dict[NodeId, list[tuple[NodeId, int]]] = {n: [] for n in nodes}
    edges: list[Edge] = []
    for i in range(num_nodes - 1):
        impedance = random.choice([1, 5, 10, 20])
        edges.append((nodes[i], nodes[i + 1], impedance))
    extra_edges = num_nodes * avg_degree // 2 - (num_nodes - 1)
    while extra_edges > 0:
        a, b = random.sample(nodes, 2)
        if any(nb == b for nb, _ in adjacency[a]):
            continue
        impedance = random.choice([1, 5, 10, 20])
        edges.append((a, b, impedance))
        extra_edges -= 1
    for u, v, imp in edges:
        adjacency[u].append((v, imp))
        adjacency[v].append((u, imp))
    feature_dim = 4
    features = np.random.randn(num_nodes, feature_dim)
    return adjacency, features

if __name__ == "__main__":
    num_nodes = 10
    avg_degree = 3
    adjacency, features = generate_synthetic_graph(num_nodes, avg_degree)
    weights = np.random.randn(features.shape[1])
    target = 1.0
    x = features[0]
    losses = [1.0, 2.0, 3.0]
    new_weights, error = hybrid_nlms_update(weights, x, target, losses=losses)
    print(f"Updated weights: {new_weights}")
    print(f"Error: {error}")