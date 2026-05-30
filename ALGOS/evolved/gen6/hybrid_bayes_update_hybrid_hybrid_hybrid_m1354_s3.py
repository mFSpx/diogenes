# DARWIN HAMMER — match 1354, survivor 3
# gen: 6
# parent_a: bayes_update.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_endpoi_m403_s0.py (gen5)
# born: 2026-05-29T23:35:30Z

"""Hybrid Bayesian-Bandit Fusion Module.

This module fuses the Bayesian evidence update primitives from *bayes_update.py* with the
contextual bandit machinery from *hybrid_hybrid_hybrid_bandit_hybrid_hybrid_endpoi_m403_s0.py*.
The mathematical bridge is built on the observation that a Bayesian posterior
\(P(H|E)=\frac{P(E|H)P(H)}{P(E)}\) can serve as a *propensity* score for a bandit
action, while the marginal likelihood \(P(E)=P(E|H)P(H)+P(\neg H)(1-P(H))\) naturally
acts as a *confidence bound* for the action. 

Thus:
* For each action we keep a Bayesian prior \(p_a\) (initially the empirical mean reward).
* The likelihood is the estimated reward probability (clipped to \([0,1]\)).
* The false‑positive rate is \(1-\text{likelihood}\).
* The resulting posterior becomes the action’s propensity.
* The marginal likelihood becomes the confidence bound returned to the bandit
  selector.

The three core functions below demonstrate this hybrid operation:
`bayes_marginal`, `bayes_update`, and `hybrid_select_action` (which internally
uses the Bayesian quantities as bandit inputs). Additional helpers manage policy
state and Bayesian parameters. """

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bayesian primitives
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Return the marginal probability P(E) = L*prior + FP*(1-prior)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Return the posterior probability P(H|E) = prior*likelihood / marginal."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Parent B – Bandit data structures and policy store
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # Bayesian posterior used as propensity
    expected_reward: float     # Empirical mean reward
    confidence_bound: float    # Bayesian marginal used as confidence bound
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float               # raw reward (assumed in [0,1])
    propensity: float           # optional, can be ignored by policy


# Global mutable stores (mirroring the original implementation)
_POLICY: Dict[str, Tuple[float, float]] = {}   # action_id -> (total_reward, count)
_BAYES_PRIOR: Dict[str, float] = {}           # action_id -> prior probability (beta mean)


def reset_policy() -> None:
    """Clear all stored statistics."""
    _POLICY.clear()
    _BAYES_PRIOR.clear()


def _empirical_reward(action: str) -> float:
    """Return the empirical mean reward for an action, or 0 if unseen."""
    total, count = _POLICY.get(action, (0.0, 0.0))
    return total / count if count > 0 else 0.0


def _ensure_prior(action: str) -> float:
    """Return a prior for the action; initialise to 0.5 if absent."""
    if action not in _BAYES_PRIOR:
        _BAYES_PRIOR[action] = 0.5
    return _BAYES_PRIOR[action]


def update_policy(updates: List[BanditUpdate]) -> None:
    """
    Incorporate a batch of observed rewards into the policy store.
    In addition to the simple reward aggregation, we also update the Bayesian
    prior for each action using the posterior from the latest observation.
    """
    for u in updates:
        # ---- simple reward aggregation (original bandit behaviour) ----
        total, count = _POLICY.get(u.action_id, (0.0, 0.0))
        _POLICY[u.action_id] = (total + float(u.reward), count + 1.0)

        # ---- Bayesian prior update (our hybrid extension) ----
        prior = _ensure_prior(u.action_id)
        # Likelihood is the observed reward (clipped to [0,1])
        likelihood = max(0.0, min(1.0, float(u.reward)))
        false_positive = 1.0 - likelihood
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        _BAYES_PRIOR[u.action_id] = posterior  # store posterior as new prior


def hybrid_select_action(
    context: Dict[str, Any],
    actions: List[str],
    algorithm: str = 'epsilon_greedy',
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a bandit strategy whose propensity scores are
    Bayesian posteriors. The confidence bound returned to the caller is the
    marginal likelihood of the corresponding Bayesian update.
    """
    if not actions:
        raise ValueError('At least one action must be provided')
    rng = random.Random(seed)

    # Compute Bayesian quantities for every candidate action
    propensities: Dict[str, float] = {}
    confidence_bounds: Dict[str, float] = {}
    expected_rewards: Dict[str, float] = {}

    for a in actions:
        prior = _ensure_prior(a)
        likelihood = _empirical_reward(a)          # treat mean reward as likelihood
        likelihood = max(0.0, min(1.0, likelihood))
        false_positive = 1.0 - likelihood
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)

        propensities[a] = posterior
        confidence_bounds[a] = marginal
        expected_rewards[a] = _empirical_reward(a)

    # ----- epsilon‑greedy selection on propensities -----
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen_id = rng.choice(actions)
    else:
        # greedy choice: highest posterior (propensity)
        chosen_id = max(actions, key=lambda a: propensities[a])

    return BanditAction(
        action_id=chosen_id,
        propensity=propensities[chosen_id],
        expected_reward=expected_rewards[chosen_id],
        confidence_bound=confidence_bounds[chosen_id],
        algorithm=algorithm,
    )


def hybrid_step(
    context: Dict[str, Any],
    actions: List[str],
    observed_updates: List[BanditUpdate],
    algorithm: str = 'epsilon_greedy',
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Perform a full hybrid iteration:
    1. Incorporate observed rewards via `update_policy`.
    2. Select the next action using `hybrid_select_action`.
    Returns the selected `BanditAction`.
    """
    update_policy(observed_updates)
    return hybrid_select_action(context, actions, algorithm, epsilon, seed)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic test to ensure no runtime errors.
    reset_policy()
    actions = ["a", "b", "c"]
    ctx = {"user": "test"}

    # Simulate a few rounds of interaction
    for round_idx in range(5):
        # No prior observations yet – all propensities start at 0.5
        chosen = hybrid_select_action(ctx, actions, algorithm='epsilon_greedy', epsilon=0.2, seed=round_idx)
        print(f"Round {round_idx}: selected {chosen.action_id} (propensity={chosen.propensity:.3f}, "
              f"expected_reward={chosen.expected_reward:.3f}, confidence={chosen.confidence_bound:.3f})")

        # Fake reward: 1 for action 'a', 0 for others (just for demonstration)
        reward = 1.0 if chosen.action_id == "a" else 0.0
        upd = BanditUpdate(context_id=f"r{round_idx}", action_id=chosen.action_id,
                           reward=reward, propensity=chosen.propensity)
        # Perform a hybrid step that updates and selects the next action
        hybrid_step(ctx, actions, [upd], algorithm='epsilon_greedy', epsilon=0.2, seed=round_idx + 100)

    print("Smoke test completed without errors.")