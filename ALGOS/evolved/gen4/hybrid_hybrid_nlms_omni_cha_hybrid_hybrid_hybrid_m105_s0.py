# DARWIN HAMMER — match 105, survivor 0
# gen: 4
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s3.py (gen3)
# born: 2026-05-29T23:26:55Z

# DARWIN HAMMER — match 14, survivor 3
# gen: 1
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s3.py (gen0)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s3.py (gen2)
# born: 2026-05-30T01:34:17Z

"""
Hybrid NLMS & Epistemic-Certainty Edge Weights

Parents
-------
* **hybrid_nlms_omni_chaotic_sprint_m59_s3.py** – implements Normalized Least Mean Squares (NLMS) algorithm.
* **hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s3.py** – builds a minimum-cost spanning tree with epistemic-certainty influenced edge weights.

Mathematical Bridge
-------------------
The key insight is to combine the NLMS prediction error with the epistemic certainty factor to obtain an effective edge weight. This is achieved by using the NLMS error as a proxy for the likelihood of error in the epistemic certainty calculation.

Given an edge e = (i, j) with physical cost d(i,j) and epistemic certainty factor c(e), we compute the epistemic certainty influenced edge weight as follows:

weight = d(i,j) * (1 - marginal) + ε

where marginal is the Bayesian-inspired marginalization of the prior, likelihood, and false-positive term:

marginal = bayes_marginal(prior, lik, fp)

The prior is computed as the normalized sum of the NLMS decision scores:

prior = (NLMS decision score(i) + NLMS decision score(j)) / (NLMS decision score(i) + NLMS decision score(j) + ε)

The likelihood is 1 - c(e), and the false-positive term is c(e) * 0.1.

This effectively combines the NLMS algorithm with the epistemic certainty influenced edge weights from the second parent.

"""

from __future__ import annotations

import math
import random
import sys
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core NLMS utilities (Algorithm A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
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

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error `target – w·x`.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


# ----------------------------------------------------------------------
# Epistemic certainty influenced edge weights (part of Algorithm B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, lik: float, fp: float) -> float:
    """Compute the Bayesian-inspired marginalization."""
    return prior * lik * (1 - fp)


def compute_epistemic_certainty_influenced_edge_weight(
    d: float, c: float, prior: float
) -> float:
    """Compute the epistemic certainty influenced edge weight."""
    marginal = bayes_marginal(prior, 1 - c, c * 0.1)
    return d * (1 - marginal) + 1e-9


# ----------------------------------------------------------------------
# Hybrid NLMS & Epistemic-Certainty Edge Weights
# ----------------------------------------------------------------------
def hybrid_nlms_epistemic_certainty(
    weights: np.ndarray, x: np.ndarray, target: float, c: float, d: float
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update with epistemic certainty influenced edge weight.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    c : float
        Epistemic certainty factor.
    d : float
        Physical cost.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error `target – w·x`.
    """
    nlms_weights, error = nlms_update(weights, x, target)
    prior = (nlms_predict(weights, x) + nlms_predict(nlms_weights, x)) / (
        nlms_predict(weights, x) + nlms_predict(nlms_weights, x) + 1e-9
    )
    weight = compute_epistemic_certainty_influenced_edge_weight(d, c, prior)
    return nlms_weights, weight


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    weights = np.random.rand(10)
    x = np.random.rand(10)
    target = 1.0
    c = 0.5
    d = 2.0
    new_weights, weight = hybrid_nlms_epistemic_certainty(weights, x, target, c, d)
    print(f"New weights: {new_weights}")
    print(f"Weight: {weight}")