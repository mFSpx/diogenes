# DARWIN HAMMER — match 1, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0.py (gen2)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s5.py (gen1)
# born: 2026-05-29T23:26:25Z

"""
This module represents a hybrid algorithm, fusing the principles of 
hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0 and 
hybrid_nlms_omni_chaotic_sprint_m59_s5. The mathematical bridge 
between these two systems is established by incorporating the 
epistemic certainty flags into the NLMS (Normalized Least Mean Squares) 
update process. The epistemic certainty flags are used to modify the 
learning rate in the NLMS update, effectively allowing the system 
to adapt and re-weight its updates based on both physical distances 
and epistemic certainty.

The core idea is to use the epistemic certainty flags to modify the 
weights in the NLMS update function, thus creating a dynamic system 
where the NLMS update and the epistemic certainty inform each other.
"""

import math
import numpy as np
import random
import sys
import pathlib

Point = tuple[float, float]
Edge = tuple[str, str]

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: tuple[str, ...] = (),
) -> dict[str, str]:
    return {
        "label": label,
        "confidence_bps": str(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
    return float(weights @ x)

def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a *batch* NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    # Predictions and errors
    preds = X @ weights
    errors = targets - preds

    # Normalized step for each sample
    powers = np.sum(X * X, axis=1) + eps  # shape (N,)
    steps = (mu * errors / powers)[:, None] * X   # shape (N, d)

    # Aggregate the per‑sample steps
    delta_w = steps.sum(axis=0)
    new_weights = weights + delta_w
    return new_weights, errors

def hybrid_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    certainty_flags: dict[Edge, dict[str, str]],
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a *batch* hybrid update, incorporating epistemic certainty.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    certainty_flags : dict[Edge, dict[str, str]]
        Epistemic certainty flags for each edge.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    # Map certainty flags to learning rate modifiers
    certainty_modifiers = {
        "FACT": 1.0,
        "PROBABLE": 0.8,
        "POSSIBLE": 0.6,
        "BULLSHIT": 0.4,
        "SURE_MAYBE": 0.2,
    }

    # Compute effective learning rate
    effective_mu = mu
    for edge, flag in certainty_flags.items():
        effective_mu *= certainty_modifiers[flag["label"]]

    # Perform NLMS update with effective learning rate
    new_weights, errors = nlms_batch_update(weights, X, targets, effective_mu, eps)
    return new_weights, errors

def hybrid_tree_cost(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    prior_probabilities: dict[str, float],
    likelihoods: dict[Edge, float],
    false_positives: dict[Edge, float],
    certainty_flags: dict[Edge, dict[str, str]],
):
    # Compute marginal probabilities
    marginals = {}
    for edge in edges:
        marginals[edge] = bayes_marginal(
            prior_probabilities[edge[0]],
            likelihoods[edge],
            false_positives[edge],
        )

    # Perform Bayesian updates
    updated_probabilities = {}
    for edge in edges:
        updated_probabilities[edge] = bayes_update(
            prior_probabilities[edge[0]],
            likelihoods[edge],
            marginals[edge],
        )

    # Compute hybrid cost
    hybrid_cost = 0.0
    for edge in edges:
        # Compute NLMS prediction error
        weights = np.array([1.0, 1.0])  # dummy weights
        X = np.array([[1.0, 2.0]])  # dummy input
        targets = np.array([3.0])  # dummy target
        _, errors = nlms_batch_update(weights, X, targets)

        # Incorporate epistemic certainty
        certainty_modifier = {
            "FACT": 1.0,
            "PROBABLE": 0.8,
            "POSSIBLE": 0.6,
            "BULLSHIT": 0.4,
            "SURE_MAYBE": 0.2,
        }[certainty_flags[edge]["label"]]
        hybrid_cost += errors[0] * certainty_modifier

    return hybrid_cost

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0.0, 0.0), "B": (1.0, 1.0)}
    edges = [("A", "B")]
    root = "A"
    prior_probabilities = {"A": 0.5}
    likelihoods = {("A", "B"): 0.8}
    false_positives = {("A", "B"): 0.2}
    certainty_flags = {("A", "B"): certainty("FACT", confidence_bps=10, authority_class="high", rationale="good")}

    hybrid_cost = hybrid_tree_cost(nodes, edges, root, prior_probabilities, likelihoods, false_positives, certainty_flags)
    print(hybrid_cost)