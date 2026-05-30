# DARWIN HAMMER — match 2906, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m2121_s3.py (gen6)
# parent_b: hybrid_bandit_router_hybrid_hybrid_hybrid_m2649_s3.py (gen5)
# born: 2026-05-29T23:46:42Z

"""Hybrid Fisher‑Bandit Algorithm
================================

This module fuses the core ideas of two parent algorithms:

* **Parent A** – computes a Fisher‑information matrix from a stylometric
  feature matrix and uses it to quantify uncertainty in dimensionality
  reduction.
* **Parent B** – implements a contextual multi‑armed bandit whose expected
  rewards are augmented by a Radial‑Basis‑Function (RBF) surrogate model.

**Mathematical bridge** – The stylometric feature vectors are used as the
context vectors for the bandit actions.  The Fisher‑information matrix
provides an uncertainty term `u = vᵀ F v` for a context vector `v`, which
is subtracted from the surrogate‑augmented reward estimate.  Thus each
action’s score is


score = empirical_reward
        + α·RBF_surrogate(v)
        – β·(vᵀ F v)               (1)


where `α,β` are tunable scalars.  Equation (1) mathematically couples the two
parents: the RBF surrogate (Parent B) operates on the same feature space
that yields the Fisher matrix (Parent A), and the Fisher matrix supplies a
principled uncertainty penalty for the bandit decision.

The implementation below provides three public functions that showcase the
hybrid operation:
`compute_fisher_information`, `update_hybrid_policy`, and `select_hybrid_action`.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = List[float]

# ----------------------------------------------------------------------
# Bandit core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridFisherBandit"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Global stores (mirroring Parent B)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_SURROGATE = None                             # will hold an RBFSurrogate instance
_FISHER_MATRIX: np.ndarray = None             # Fisher‑information matrix (Parent A)


# ----------------------------------------------------------------------
# RBF surrogate (identical to Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel exp(‑(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class RBFSurrogate:
    """Simple RBF surrogate model."""

    def __init__(self, centers: List[Vector], weights: List[float], epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def evaluate(self, x: Vector) -> float:
        """Weighted sum of Gaussian kernels centred at stored points."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

    def add_center(self, x: Vector, weight: float) -> None:
        """Append a new centre‑weight pair."""
        self.centers.append(x)
        self.weights.append(weight)


# ----------------------------------------------------------------------
# Fisher‑information utilities (derived from Parent A)
# ----------------------------------------------------------------------
def compute_fisher_information(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Approximate the Fisher‑information matrix for a set of feature vectors.

    The Fisher matrix is taken as the (pseudo‑)inverse of the empirical
    covariance of the feature matrix.  This provides a measure of
    information density in each direction of the feature space.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Fisher‑information matrix of shape (n_features, n_features).
    """
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2‑dimensional")
    # Center the data
    centered = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    # Empirical covariance (bias‑corrected)
    cov = np.cov(centered, rowvar=False, bias=False)
    # Pseudo‑inverse to handle singular covariances
    fisher = np.linalg.pinv(cov)
    return fisher


def uncertainty_estimate(context_vec: Vector, fisher_matrix: np.ndarray) -> float:
    """
    Quadratic form vᵀ F v giving an uncertainty estimate for a context vector.

    Parameters
    ----------
    context_vec : Vector
        Feature vector (length = n_features).
    fisher_matrix : np.ndarray
        Fisher‑information matrix of shape (n_features, n_features).

    Returns
    -------
    float
        Non‑negative uncertainty value.
    """
    v = np.asarray(context_vec, dtype=float)
    return float(v @ fisher_matrix @ v)


# ----------------------------------------------------------------------
# Hybrid policy management
# ----------------------------------------------------------------------
def reset_hybrid_state() -> None:
    """Clear policy, surrogate, and Fisher matrix."""
    _POLICY.clear()
    global _SURROGATE, _FISHER_MATRIX
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)
    _FISHER_MATRIX = None


def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def update_hybrid_policy(
    update: BanditUpdate,
    context_vector: Vector,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> None:
    """
    Incorporate a new reward observation into both the empirical bandit
    statistics and the RBF surrogate, using the Fisher‑information matrix
    for uncertainty weighting.

    Parameters
    ----------
    update : BanditUpdate
        The observed reward and associated metadata.
    context_vector : Vector
        Stylometric feature vector that serves as the surrogate input.
    alpha, beta : float
        Scaling factors for surrogate contribution and Fisher penalty.
    """
    # ----- Empirical bandit update -----
    total, n = _POLICY.get(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id] = [total + update.reward, n + 1]

    # ----- Ensure Fisher matrix is available -----
    global _FISHER_MATRIX
    if _FISHER_MATRIX is None:
        # Lazy initialisation with a single‑sample placeholder
        placeholder = np.atleast_2d(context_vector)
        _FISHER_MATRIX = compute_fisher_information(placeholder)

    # ----- Surrogate update -----
    # Weight the new centre by the reward adjusted for uncertainty
    unc = uncertainty_estimate(context_vector, _FISHER_MATRIX)
    surrogate_weight = alpha * (update.reward - beta * unc)
    _SURROGATE.add_center(context_vector, surrogate_weight)


def select_hybrid_action(
    context_vector: Vector,
    candidate_actions: List[str],
    alpha: float = 1.0,
    beta: float = 0.5,
) -> BanditAction:
    """
    Choose the best action for a given context by scoring each candidate
    with the hybrid objective (1).

    Parameters
    ----------
    context_vector : Vector
        Feature representation of the current context.
    candidate_actions : list[str]
        Identifiers of actions that can be taken.
    alpha, beta : float
        Same scaling constants used in `update_hybrid_policy`.

    Returns
    -------
    BanditAction
        The action with the highest hybrid score.
    """
    if not candidate_actions:
        raise ValueError("candidate_actions must contain at least one action")

    # Ensure the surrogate and Fisher matrix exist
    global _FISHER_MATRIX
    if _FISHER_MATRIX is None:
        _FISHER_MATRIX = compute_fisher_information(np.atleast_2d(context_vector))

    surrogate_val = _SURROGATE.evaluate(context_vector)
    unc = uncertainty_estimate(context_vector, _FISHER_MATRIX)

    best_score = -math.inf
    best_action_id = None
    for a_id in candidate_actions:
        emp = _empirical_reward(a_id)
        # Hybrid score per equation (1)
        score = emp + alpha * surrogate_val - beta * unc
        if score > best_score:
            best_score = score
            best_action_id = a_id

    # Propensity is a softmax‑like probability; for simplicity we use a
    # normalized exponential over the scores of the candidates.
    scores = np.array(
        [
            _empirical_reward(a) + alpha * surrogate_val - beta * unc
            for a in candidate_actions
        ]
    )
    exp_scores = np.exp(scores - np.max(scores))  # stability
    probs = exp_scores / exp_scores.sum()
    propensity = float(probs[candidate_actions.index(best_action_id)])

    # Confidence bound (simple UCB‑style)
    total_counts = sum(_POLICY.get(a, [0.0, 0.0])[1] for a in candidate_actions) + 1e-9
    count_best = _POLICY.get(best_action_id, [0.0, 0.0])[1] + 1e-9
    confidence = math.sqrt(2 * math.log(total_counts) / count_best)

    return BanditAction(
        action_id=best_action_id,
        propensity=propensity,
        expected_reward=_empirical_reward(best_action_id),
        confidence_bound=confidence,
    )


# ----------------------------------------------------------------------
# Example utilities (demonstrate the hybrid pipeline)
# ----------------------------------------------------------------------
def generate_dummy_features(n_samples: int, n_features: int) -> np.ndarray:
    """Create a random feature matrix mimicking stylometric vectors."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))


def demo_hybrid_cycle() -> None:
    """Run a short demonstration of the hybrid algorithm."""
    reset_hybrid_state()

    # 1️⃣ Build a stylometric feature matrix and compute Fisher info
    feats = generate_dummy_features(n_samples=10, n_features=5)
    global _FISHER_MATRIX
    _FISHER_MATRIX = compute_fisher_information(feats)

    # 2️⃣ Define a set of actions
    actions = ["A", "B", "C"]
    for a in actions:
        _POLICY[a] = [0.0, 0.0]  # initialise

    # 3️⃣ Simulate a few interaction steps
    for step in range(5):
        # Randomly pick a context vector from the matrix
        ctx_vec = feats[step % len(feats)].tolist()

        # Choose an action using the hybrid selector
        chosen = select_hybrid_action(ctx_vec, actions)
        print(f"Step {step}: selected {chosen.action_id} (propensity={chosen.propensity:.3f})")

        # Simulate a stochastic reward (higher for action 'B' as an example)
        true_means = {"A": 0.2, "B": 0.8, "C": 0.5}
        reward = random.gauss(true_means[chosen.action_id], 0.1)

        # Feed the observation back into the hybrid policy
        upd = BanditUpdate(
            context_id=f"ctx_{step}",
            action_id=chosen.action_id,
            reward=reward,
            propensity=chosen.propensity,
        )
        update_hybrid_policy(upd, ctx_vec)

    print("\nFinal policy statistics:")
    for a, (tot, cnt) in _POLICY.items():
        print(f"  Action {a}: total_reward={tot:.3f}, count={cnt}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_cycle()