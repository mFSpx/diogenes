# DARWIN HAMMER — match 2163, survivor 3
# gen: 5
# parent_a: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m24_s0.py (gen4)
# born: 2026-05-29T23:41:05Z

"""Hybrid Tropical‑Bayesian Tree + Bandit‑Capybara SSIM Decision

Parents
-------
* **tropical_maxplus.py** – defines the max‑plus semiring (⊕ = max,
  ⊗ = +) and tropical matrix utilities.
* **hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m24_s0.py** – provides a
  resource‑store dynamics (Bandit‑Capybara scheduler‑optimizer) together
  with a Structural Similarity Index (SSIM) based weighting for a
  decision‑hygiene score.

Mathematical Bridge
-------------------
Both parents operate on log‑probabilities.  In the tropical semiring a
product of probabilities corresponds to an addition (⊗), while the
maximum‑likelihood marginalisation is a tropical addition (⊕ = max).
The Bandit‑Capybara side treats Euclidean edge costs as negative
log‑likelihoods and updates a store variable that can be interpreted as
a log‑probability mass.  By feeding the tropical belief propagation
output into the store update and then weighting the resulting decision
score with an SSIM similarity between feature vectors, we obtain a
single unified system that simultaneously performs:

1. Tropical belief propagation on a tree (max‑plus algebra).
2. Store‑based resource dynamics driven by the propagated beliefs.
3. SSIM‑scaled decision hygiene scoring.

The code below implements this fusion while preserving the original
interfaces where possible.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tropical max‑plus primitives (Parent A)
# ---------------------------------------------------------------------------

def t_add(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Tropical addition (⊕): max(x, y). Works element‑wise."""
    return np.maximum(x, y)


def t_mul(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Tropical multiplication (⊗): x + y. Works element‑wise."""
    return np.add(x, y)


def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication.

    C[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("Incompatible shapes for tropical matmul")
    # Broadcast addition and take max over k
    C = np.full((m, n), -np.inf, dtype=float)
    for k in range(p):
        C = t_add(C, A[:, k:k+1] + B[k:k+1, :])
    return C


# ---------------------------------------------------------------------------
# Resource‑store dynamics & SSIM utilities (Parent B)
# ---------------------------------------------------------------------------

# Constants (taken from parent B)
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step
ETA0 = 0.01          # base learning rate
DELTA_MAX = 1.0      # maximum evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule

# ---------------------------------------------------------------------------
# Helper functions from Parent B
# ---------------------------------------------------------------------------

def update_store(store: float, inflow: float, outflow: float) -> float:
    """Euler update of the scalar store variable."""
    return store + DT * (ALPHA * inflow - BETA * outflow)


def update_weight_matrix(W: np.ndarray, grad: np.ndarray, epoch: int) -> np.ndarray:
    """
    Simple gradient descent on a weight matrix.

    Learning rate decays as ETA0 / (1 + epoch).
    """
    eta = ETA0 / (1 + epoch)
    return W - eta * grad


def evasion_perturb(position: np.ndarray, epoch: int) -> np.ndarray:
    """
    Apply a bounded random perturbation whose magnitude decays with epoch.
    """
    magnitude = DELTA_MAX * math.exp(-ALPHA_EVASION * epoch)
    direction = np.random.randn(*position.shape)
    direction /= np.linalg.norm(direction) + 1e-12
    return position + magnitude * direction


def ssim_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Simplified SSIM for 1‑D feature vectors.

    Returns a value in [0, 1] where 1 means identical.
    """
    C1 = 1e-4
    C2 = 9e-4
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    cov_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / (denominator + 1e-12))


# ---------------------------------------------------------------------------
# Hybrid core – integrates both parent behaviours
# ---------------------------------------------------------------------------

def hybrid_belief_propagation(
    log_prior: np.ndarray,
    edge_costs: np.ndarray,
) -> np.ndarray:
    """
    Propagate log‑probabilities through a tree using tropical matrix
    multiplication.  Edge costs are interpreted as negative log‑likelihoods,
    therefore we add them (tropical multiplication) to the priors and take
    the tropical max‑plus product.
    """
    # Convert edge costs to tropical “multiplication” by negating (since they
    # are negative log‑likelihoods).
    neg_costs = -edge_costs
    return t_matmul(log_prior, neg_costs)


def hybrid_store_and_weights(
    belief: np.ndarray,
    store: float,
    W: np.ndarray,
    position: np.ndarray,
    epoch: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Update the scalar store, weight matrix, and position based on the current
    belief vector.
    """
    # Inflow is the total positive belief mass, outflow the negative part.
    inflow = float(np.sum(np.maximum(belief, 0)))
    outflow = float(np.sum(np.maximum(-belief, 0)))

    new_store = update_store(store, inflow, outflow)

    # Dummy gradient: outer product of belief with itself (captures
    # correlation of belief components).
    grad = np.outer(belief, belief)
    new_W = update_weight_matrix(W, grad, epoch)

    new_position = evasion_perturb(position, epoch)

    return new_store, new_W, new_position


def hybrid_decision_score(
    belief: np.ndarray,
    feature_vec_a: np.ndarray,
    feature_vec_b: np.ndarray,
    position: np.ndarray,
) -> float:
    """
    Compute a decision‑hygiene score by weighting the summed belief with an
    SSIM similarity between two feature vectors and penalising large
    positional displacement.
    """
    belief_sum = float(np.sum(belief))
    similarity = ssim_score(feature_vec_a, feature_vec_b)
    penalty = np.linalg.norm(position)
    return similarity * belief_sum - penalty


def hybrid_step(
    log_prior: np.ndarray,
    edge_costs: np.ndarray,
    store: float,
    W: np.ndarray,
    position: np.ndarray,
    feature_vec_a: np.ndarray,
    feature_vec_b: np.ndarray,
    epoch: int,
) -> Dict[str, Any]:
    """
    One full hybrid iteration:
      1. Tropical belief propagation.
      2. Store / weight / position dynamics.
      3. SSIM‑scaled decision hygiene score.

    Returns a dictionary with all intermediate results.
    """
    belief = hybrid_belief_propagation(log_prior, edge_costs)
    new_store, new_W, new_position = hybrid_store_and_weights(
        belief, store, W, position, epoch
    )
    decision = hybrid_decision_score(belief, feature_vec_a, feature_vec_b, new_position)

    return {
        "belief": belief,
        "store": new_store,
        "weight_matrix": new_W,
        "position": new_position,
        "decision_score": decision,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Small synthetic example
    rng = np.random.default_rng(42)

    # Log‑priors for 4 nodes (row vector)
    log_prior = np.log(rng.random((1, 4)) + 1e-6)

    # Edge costs between nodes (4×4 matrix, symmetric, zero diagonal)
    edge_costs = rng.random((4, 4))
    np.fill_diagonal(edge_costs, 0.0)

    # Initial store, weight matrix and position
    store = 1.0
    W = np.eye(4)
    position = rng.random(4)

    # Feature vectors for SSIM (length 9 as in parent B)
    feature_vec_a = rng.random(9)
    feature_vec_b = rng.random(9)

    epoch = 0
    result = hybrid_step(
        log_prior=log_prior,
        edge_costs=edge_costs,
        store=store,
        W=W,
        position=position,
        feature_vec_a=feature_vec_a,
        feature_vec_b=feature_vec_b,
        epoch=epoch,
    )

    print("Hybrid step result keys:", list(result.keys()))
    print("Decision score:", result["decision_score"])
    # Ensure no NaNs
    assert not np.isnan(result["decision_score"]), "Decision score is NaN"
    print("Smoke test passed.")