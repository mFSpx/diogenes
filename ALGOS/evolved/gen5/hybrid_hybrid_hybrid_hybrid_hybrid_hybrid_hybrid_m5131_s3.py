# DARWIN HAMMER — match 5131, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_bandit_m1620_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m936_s1.py (gen4)
# born: 2026-05-30T00:00:07Z

"""Hybrid Bandit‑Koopman‑Curvature Fusion Module
================================================

This module merges the core mathematics of the two parent algorithms:

* **Parent A** – a contextual multi‑armed bandit whose selected action’s
  ``propensity`` and ``expected_reward`` are used to weight graph‑theoretic
  computations (e.g. Ollivier‑Ricci curvature).

* **Parent B** – a Koopman‑operator style dynamical update together with a
  Bayesian‐hybrid belief update and a ternary router evaluation.

**Mathematical bridge**

The bridge is the *propensity* (or confidence) of the chosen bandit action.
We treat this scalar as a multiplicative factor for the dynamical updates
(Koopman and hybrid belief) and as a node‑weight in the curvature‑like
metric.  Concretely:

* ``koopman_bandit_update`` :  
  ``observable_{t+1} = observable_t + 0.1 * propensity * observation``

* ``hybrid_belief_update`` :  
  Evidence and observation are scaled by ``propensity`` before being fed to
  the Bayesian update from Parent B.

* ``weighted_curvature`` :  
  Given an adjacency matrix ``A`` and a node‑weight vector derived from the
  bandit action (``propensity`` distributed over the nodes), we compute a
  weighted Laplacian ``L_w = D_w - A`` and return a simple curvature proxy
  ``trace(L_w) / sum(weights)``.

All bandit bookkeeping (policy, reward estimation, action selection) is
identical to the parents and shared across the fused functions.

The three public functions below demonstrate the hybrid operation:
``select_action``, ``koopman_bandit_update`` and ``weighted_curvature``.
"""

import os
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Global policy store (shared by all bandit‑related functions)
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear the global bandit policy."""
    _POLICY.clear()

def update_policy(updates: list["BanditUpdate"]) -> None:
    """Incorporate a batch of bandit feedback into the policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)          # cumulative reward
        stats[1] += 1.0                       # count of pulls

def _reward(action: str) -> float:
    """Mean reward observed for a given action."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Number of times an action has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

# ----------------------------------------------------------------------
# Data structures (identical to both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def select_action(
    context: dict[str, float],
    actions: list[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Contextual action selection.

    The implementation follows Parent A (epsilon‑greedy, Thompson,
    LinUCB‑style).  The returned ``propensity`` is the softmax of the
    estimated rewards and serves as the scaling factor for downstream
    dynamical updates.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # ----- exploration / exploitation choice -----
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Sample from a beta posterior built from reward statistics
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))
            ),
        )
    else:
        # LinUCB‑style: compute an upper confidence bound
        # For simplicity we use reward mean + sqrt(count) as a proxy.
        ucb = {}
        for a in actions:
            mean = _reward(a)
            cnt = _count(a)
            bonus = math.sqrt(math.log(max(1, len(_POLICY) + 1)) / (cnt + 1e-6))
            ucb[a] = mean + bonus
        chosen = max(ucb, key=ucb.get)

    # Propensity via softmax over estimated rewards
    rewards = np.array([_reward(a) for a in actions])
    max_r = np.max(rewards)
    exp_vals = np.exp(rewards - max_r)  # stability
    probs = exp_vals / exp_vals.sum()
    propensity = float(probs[actions.index(chosen)])

    expected_reward = _reward(chosen)
    confidence_bound = float(math.sqrt(1.0 / (1.0 + _count(chosen))))

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected_reward,
        confidence_bound=confidence_bound,
        algorithm=algorithm,
    )


def koopman_bandit_update(
    observable: float,
    observation: float,
    action: BanditAction,
) -> float:
    """
    Koopman‑style dynamical update modulated by the bandit propensity.

    Parent B defines ``observable_{t+1} = observable_t + 0.1 * observation``.
    Here we multiply the observation term by the action's ``propensity``,
    thereby linking reinforcement feedback to the system dynamics.
    """
    scale = 0.1 * action.propensity
    return observable + scale * observation


def hybrid_belief_update(
    hypothesis: np.ndarray,
    evidence: np.ndarray,
    observation: np.ndarray,
    action: BanditAction,
) -> np.ndarray:
    """
    Bayesian hybrid belief update where both evidence and observation are
    weighted by the bandit propensity.  The algebra mirrors Parent B's
    ``hybrid_update`` but inserts the scaling factor.
    """
    # Scale matrices/vectors
    prop = action.propensity
    scaled_evidence = evidence * prop
    scaled_observation = observation * prop

    # Compute likelihood ratio (identical to Parent B)
    # Using the same expression for compatibility.
    likelihood_ratio = math.exp(
        -0.5 * np.dot(scaled_evidence.T, np.dot(hypothesis, hypothesis.T))
        - 0.5 * np.dot(scaled_observation.T, np.dot(hypothesis, hypothesis.T))
    )

    # Posterior covariance (regularized)
    var = np.linalg.inv(np.dot(hypothesis.T, hypothesis) + np.eye(hypothesis.shape[0]))

    # Updated mean (same as Parent B's helper)
    updated_mean = hypothesis + np.dot(var, scaled_observation)

    # Return a tuple that contains both the likelihood ratio and the new mean
    # for downstream use.
    return np.concatenate([np.array([likelihood_ratio]), updated_mean.ravel()])


def weighted_curvature(
    adjacency: np.ndarray,
    action: BanditAction,
    node_feature: np.ndarray | None = None,
) -> float:
    """
    Compute a curvature‑like scalar on a graph where node weights are derived
    from the bandit action.

    - ``adjacency`` : symmetric binary (or weighted) adjacency matrix.
    - ``action.propensity`` is spread uniformly over all nodes unless a
      ``node_feature`` vector is supplied; in that case the feature vector is
      normalized and multiplied by the propensity to obtain heterogeneous
      weights.
    - The weighted degree matrix ``D_w`` is constructed from these weights,
      and the weighted Laplacian ``L_w = D_w - A`` is formed.
    - A simple curvature proxy is ``trace(L_w) / sum(weights)`` which reduces
      to the average node degree when all weights are equal.

    This mirrors Parent A’s intention to let bandit context influence
    curvature computation while staying fully differentiable with NumPy.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square")

    n = adjacency.shape[0]

    if node_feature is not None:
        if node_feature.shape[0] != n:
            raise ValueError("node_feature length must match number of nodes")
        # Normalize feature to sum to 1, then scale by propensity
        weights = node_feature.astype(float)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / n
        else:
            weights = weights / weights.sum()
        weights = weights * action.propensity
    else:
        # Uniform weights scaled by propensity
        weights = np.full(n, action.propensity / n)

    # Weighted degree matrix
    D_w = np.diag(adjacency @ weights)

    # Weighted Laplacian
    L_w = D_w - adjacency

    curvature = np.trace(L_w) / (weights.sum() + 1e-12)
    return float(curvature)


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny policy
    reset_policy()
    dummy_updates = [
        BanditUpdate(context_id="c1", action_id="a", reward=1.0, propensity=0.5),
        BanditUpdate(context_id="c2", action_id="b", reward=0.0, propensity=0.5),
    ]
    update_policy(dummy_updates)

    # Context and actions
    ctx = {"feature1": 0.2, "feature2": 0.8}
    actions = ["a", "b", "c"]

    # Select an action
    act = select_action(context=ctx, actions=actions, algorithm="linucb")
    print("Selected action:", act)

    # Perform a Koopman‑bandit update
    obs = 3.14
    new_obs = koopman_bandit_update(observable=1.0, observation=obs, action=act)
    print("Koopman update result:", new_obs)

    # Hybrid belief update
    H = np.eye(3)
    E = np.random.rand(3, 3)
    O = np.random.rand(3)
    belief = hybrid_belief_update(H, E, O, act)
    print("Hybrid belief vector (likelihood + mean):", belief)

    # Weighted curvature on a small graph
    A = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=float)
    curv = weighted_curvature(A, act)
    print("Weighted curvature proxy:", curv)