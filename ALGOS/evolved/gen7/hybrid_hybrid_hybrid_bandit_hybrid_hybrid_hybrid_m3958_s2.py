# DARWIN HAMMER — match 3958, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m547_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2380_s1.py (gen6)
# born: 2026-05-29T23:52:49Z

"""Hybrid Algorithm: Bandit‑Router ↔ Fisher‑Localization ↔ Tropical Linear Algebra
===================================================================================

Parent A (Bandit Router) provides a contextual multi‑armed bandit model where each
action is characterised by a *propensity* (probability of being chosen) and an
estimated *expected reward*.  

Parent B (Fisher Localization + Tropical Algebra) supplies two mathematical
building blocks:
    1. Gaussian beams and their Fisher information, used to evaluate the
       information‑gain of an action.
    2. Max‑plus (tropical) matrix multiplication, a non‑linear analogue of the
       usual linear transform.

**Mathematical Bridge**

For a given context (represented by an angle ``theta``) each bandit action
``a`` is mapped to a Gaussian beam

    g_a(theta) = propensity_a · exp(-((theta‑mu_a)/σ)²)

where ``mu_a`` is a deterministic “beam centre’’ derived from the action’s
identifier (hash → angle) and ``σ`` is a shared bandwidth.  The beam amplitude
is exactly the bandit propensity, thus the bandit policy becomes a set of
Gaussian amplitudes.

The vector of beam values ``g = [g_a]`` is then processed by a tropical weight
matrix ``W`` (size |A|×|A|).  The tropical product

    τ = g ⊗ W   (max‑plus multiplication)

produces a *tropical score* for each action that captures the best‑possible
cumulative contribution of downstream actions under the max‑plus algebra.

Finally, a Fisher‑information term

    I_a(theta) = ( (theta‑mu_a) / σ² )² · g_a(theta)

quantifies the statistical usefulness of the beam.  The hybrid decision rule
selects the action maximising the product of tropical score and Fisher
information:

    a* = argmax_a   τ_a · I_a(theta)

The selected action is returned together with its updated bandit statistics.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bandit Router Core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # interpreted as Gaussian amplitude
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear the global policy statistics."""
    _POLICY.clear()


def _reward(a: str) -> float:
    """Return the average reward observed for action ``a``."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global reward statistics."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Parent B – Fisher Localization Core
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, mu: float, sigma: float, amplitude: float) -> float:
    """Gaussian beam amplitude at angle ``theta``."""
    return amplitude * math.exp(-((theta - mu) / sigma) ** 2)


def fisher_information(theta: float, mu: float, sigma: float, amplitude: float) -> float:
    """
    Simple Fisher‑information proxy for a Gaussian beam.
    I(μ) = (∂/∂μ log g)² · g  = ((θ‑μ)/σ²)² · g(θ;μ,σ,amp)
    """
    g = gaussian_beam(theta, mu, sigma, amplitude)
    if g == 0.0:
        return 0.0
    dlog = (theta - mu) / (sigma ** 2)
    return (dlog ** 2) * g


# ----------------------------------------------------------------------
# Parent B – Tropical Linear Algebra Utilities
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Max‑plus matrix multiplication.
    (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for tropical multiplication")
    # Broadcast addition then max over the shared axis
    C = np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)
    return C


def adjusted_grad_hess(logistic_loss: np.ndarray, alpha: float,
                       s: np.ndarray, H: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Element‑wise adjusted gradient and Hessian for a hybrid loss.
    """
    grad_base = logistic_loss * (1.0 - logistic_loss)
    hess_base = logistic_loss * (1.0 - logistic_loss) * (1.0 - 2.0 * logistic_loss)

    grad = grad_base + alpha * s * H
    hess = hess_base + alpha * s * H
    return grad, hess


# ----------------------------------------------------------------------
# Minimal Morphology placeholder (required for import compatibility)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float = 0.0
    width: float = 0.0
    curvature: float = 0.0


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def _action_mu(action_id: str) -> float:
    """
    Deterministic mapping from an action identifier to a beam centre (mu).
    The hash is reduced to a value in [0, 2π) for angular consistency.
    """
    # Use a stable hash (Python's built‑in hash is salted per process; we avoid it)
    # Simple deterministic conversion: sum of Unicode code points modulo 360.
    total = sum(ord(ch) for ch in action_id)
    return (total % 360) * (math.pi / 180.0)  # radians


def compute_beam_scores(actions: List[BanditAction],
                        theta: float,
                        sigma: float = 1.0) -> np.ndarray:
    """
    Convert each BanditAction into a Gaussian beam value evaluated at ``theta``.
    Returns a 1‑D array ``g`` where g[i] = propensity_i · exp(-((θ‑μ_i)/σ)²).
    """
    scores = []
    for a in actions:
        mu = _action_mu(a.action_id)
        g = gaussian_beam(theta, mu, sigma, a.propensity)
        scores.append(g)
    return np.array(scores, dtype=float)


def evaluate_tropical_policy(scores: np.ndarray,
                             weight_matrix: np.ndarray) -> np.ndarray:
    """
    Apply max‑plus multiplication to propagate beam scores through ``weight_matrix``.
    ``scores`` is interpreted as a row vector (1 × n).  The result is a 1‑D array
    of tropical scores (length = weight_matrix.shape[1]).
    """
    # Ensure 2‑D shapes for tropical_matmul
    row_vec = scores.reshape(1, -1)          # shape (1, n)
    tropical = tropical_matmul(row_vec, weight_matrix)  # shape (1, m)
    return tropical.ravel()                  # flatten to 1‑D


def select_best_action(actions: List[BanditAction],
                       theta: float,
                       sigma: float,
                       tropical_scores: np.ndarray) -> BanditAction:
    """
    Combine tropical scores with Fisher information to pick the optimal action.
    The selection criterion is:
        argmax_a  (tropical_score_a * fisher_information_a)
    """
    best_val = -math.inf
    best_action = None
    for idx, a in enumerate(actions):
        mu = _action_mu(a.action_id)
        fisher = fisher_information(theta, mu, sigma, a.propensity)
        combined = tropical_scores[idx] * fisher
        if combined > best_val:
            best_val = combined
            best_action = a
    # Fallback – should never happen because actions is non‑empty
    return best_action if best_action is not None else actions[0]


def hybrid_step(context_id: str,
                theta: float,
                actions: List[BanditAction],
                weight_matrix: np.ndarray,
                sigma: float = 1.0) -> BanditAction:
    """
    Execute one hybrid decision step:
        1. Compute Gaussian beam scores from bandit propensities.
        2. Propagate scores with tropical matrix multiplication.
        3. Select the action that maximises tropical_score × Fisher information.
        4. Return the chosen ``BanditAction`` (no side‑effects on the policy).
    """
    scores = compute_beam_scores(actions, theta, sigma)
    tropical_scores = evaluate_tropical_policy(scores, weight_matrix)
    chosen = select_best_action(actions, theta, sigma, tropical_scores)
    return chosen


def hybrid_update(updates: List[BanditUpdate],
                  weight_matrix: np.ndarray,
                  learning_rate: float = 0.1) -> np.ndarray:
    """
    Perform a policy update based on observed rewards and adjust the tropical
    weight matrix using a simple gradient‑like rule derived from the hybrid loss.

    Returns the possibly modified weight matrix.
    """
    # 1. Update the bandit reward statistics.
    update_policy(updates)

    # 2. Build a logistic‑loss‑like vector from the updated average rewards.
    rewards = np.array([_reward(u.action_id) for u in updates], dtype=float)
    logistic = 1.0 / (1.0 + np.exp(-rewards))  # sigmoid as proxy loss

    # 3. Compute a synthetic gradient/Hessian using the helper from Parent B.
    grad, hess = adjusted_grad_hess(logistic, alpha=0.5,
                                    s=np.ones_like(logistic), H=0.01)

    # 4. Adjust the tropical weight matrix (element‑wise) in the direction of
    #    the gradient.  We keep the max‑plus semantics by adding the update.
    update = learning_rate * grad[:, np.newaxis]  # broadcast over columns
    weight_matrix = weight_matrix + update
    return weight_matrix


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Define a small set of mock actions
    actions = [
        BanditAction(action_id="alpha", propensity=0.8, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="bandit"),
        BanditAction(action_id="beta", propensity=0.5, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="bandit"),
        BanditAction(action_id="gamma", propensity=0.3, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="bandit"),
    ]

    # Random tropical weight matrix (max‑plus values)
    W = np.random.rand(len(actions), len(actions)) * 2.0  # values in [0,2)

    # Contextual angle (radians)
    theta = math.radians(45.0)  # 45 degrees

    # Run a hybrid decision step
    chosen = hybrid_step(context_id="ctx1",
                         theta=theta,
                         actions=actions,
                         weight_matrix=W,
                         sigma=0.5)

    print(f"Chosen action: {chosen.action_id}")

    # Simulate an observed reward and perform an update
    reward = random.random()  # mock reward
    update = BanditUpdate(context_id="ctx1",
                          action_id=chosen.action_id,
                          reward=reward,
                          propensity=chosen.propensity)

    W = hybrid_update([update], W)

    # Verify that the policy dictionary now contains the action entry
    print(f"Policy stats for '{chosen.action_id}': {_POLICY.get(chosen.action_id)}")