# DARWIN HAMMER — match 1653, survivor 1
# gen: 6
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1016_s3.py (gen5)
# born: 2026-05-29T23:38:02Z

"""Hybrid Hoeffding‑Gini Bandit

This module fuses two previously independent algorithms:

* **Parent A** – a Hoeffding bound that is regularised by the Gini coefficient
  (used for split decisions in a Hoeffding tree).
* **Parent B** – a contextual multi‑armed bandit that stores reward statistics
  with a Count‑Min Sketch and selects actions via an Upper‑Confidence‑Bound (UCB)
  rule.

The mathematical bridge is the *Hoeffding confidence interval*.
Both algorithms need a bound on the deviation of an empirical mean from its
true mean.  In Parent A the bound is enlarged by a term proportional to the
Gini coefficient of the class distribution; in Parent B the same bound can be
applied to the estimated reward of each arm.  By feeding the Gini coefficient
computed over the reward distribution of all arms into the Hoeffding bound we
obtain a **Gini‑regularised UCB** that simultaneously accounts for reward
inequality (exploration pressure) and statistical confidence (exploitation).

The resulting hybrid can be used wherever a bandit must adapt to highly
skewed reward landscapes while still enjoying the theoretical guarantees of
Hoeffding’s inequality.
"""

import math
import random
import sys
from pathlib import Path
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Core mathematical primitives (shared by both parents)
# ----------------------------------------------------------------------


def hoeffding_bound_with_gini(r: float, delta: float, n: int, gini_coeff: float) -> float:
    """
    Hoeffding bound regularised by a Gini term.
    r      – range of the random variable (max - min)
    delta  – desired failure probability (0 < delta < 1)
    n      – number of observations
    gini_coeff – Gini coefficient of the underlying distribution
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    regularization_term = gini_coeff * math.pi / 6.0
    return math.sqrt((r * r * math.log(1.0 / delta) + regularization_term) / (2.0 * n))


def gini_coefficient(values: List[float]) -> float:
    """
    Standard Gini coefficient for a list of non‑negative numbers.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = 0.0
    for i, val in enumerate(sorted_vals, start=1):
        cumulative += i * val
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2.0 * cumulative) / (n * total) - (n + 1) / n
    return gini


# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent B)
# ----------------------------------------------------------------------


def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Hash an item into `depth` positions in a width‑wide table."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """
    Simple Count‑Min Sketch implementation.
    Returns a (depth, width) integer matrix with frequency estimates.
    """
    cms = np.zeros((depth, width), dtype=int)
    for item in items:
        hashes = _cms_hash(item, depth, width)
        for d, h in enumerate(hashes):
            cms[d, h] += 1
    return cms


def cms_estimate(cms: np.ndarray, item: str) -> int:
    """Estimate frequency of `item` using the previously built sketch."""
    depth, width = cms.shape
    hashes = _cms_hash(item, depth, width)
    return min(cms[d, h] for d, h in enumerate(hashes))


# ----------------------------------------------------------------------
# Bandit data structures (Parent B)
# ----------------------------------------------------------------------


class BanditAction:
    def __init__(
        self,
        action_id: str,
        propensity: float = 0.0,
        expected_reward: float = 0.0,
        confidence_bound: float = 0.0,
        algorithm: str = "gini_ucb",
    ):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

    def ucb_score(self) -> float:
        return self.expected_reward + self.confidence_bound


class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity


# ----------------------------------------------------------------------
# Global policy store (mirrors Parent B)
# ----------------------------------------------------------------------


_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count, total_propensity]


def reset_policy() -> None:
    _POLICY.clear()


def _policy_stats(action_id: str) -> Tuple[float, float, float]:
    """Return (total_reward, count, total_propensity) for an action."""
    return tuple(_POLICY.get(action_id, [0.0, 0.0, 0.0]))


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy using raw reward observations."""
    for u in updates:
        total, cnt, prop = _policy_stats(u.action_id)
        _POLICY[u.action_id] = [total + float(u.reward), cnt + 1.0, prop + float(u.propensity)]


def _average_reward(action_id: str) -> float:
    total, cnt, _ = _policy_stats(action_id)
    return total / cnt if cnt > 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid operations (the new fused logic)
# ----------------------------------------------------------------------


def compute_action_confidence(
    action_id: str,
    r: float,
    delta: float,
    gini_over_actions: float,
) -> float:
    """
    Compute a Gini‑regularised Hoeffding confidence bound for a single arm.
    The number of observations `n` is taken from the policy store.
    """
    _, cnt, _ = _policy_stats(action_id)
    n = int(cnt) if cnt > 0 else 1  # avoid division by zero
    return hoeffding_bound_with_gini(r, delta, n, gini_over_actions)


def refresh_actions(actions: List[BanditAction], r: float, delta: float) -> None:
    """
    Update each action's expected reward and confidence bound using the
    hybrid Gini‑regularised Hoeffding rule.
    """
    # Gather average rewards for all actions to compute a global Gini coefficient
    avg_rewards = [_average_reward(a.action_id) for a in actions]
    global_gini = gini_coefficient(avg_rewards)

    for a in actions:
        a.expected_reward = _average_reward(a.action_id)
        a.confidence_bound = compute_action_confidence(a.action_id, r, delta, global_gini)


def select_ucb_action(actions: List[BanditAction]) -> BanditAction:
    """
    Return the action with the highest Upper Confidence Bound (UCB) score.
    """
    if not actions:
        raise ValueError("Action list is empty")
    return max(actions, key=lambda a: a.ucb_score())


def hybrid_bandit_step(
    context_id: str,
    candidate_actions: List[str],
    r: float = 1.0,
    delta: float = 0.05,
) -> Tuple[BanditAction, List[BanditUpdate]]:
    """
    Perform a single decision step:
      1. Wrap candidate action ids into BanditAction objects.
      2. Refresh their statistics using the hybrid confidence bound.
      3. Choose the best action via UCB.
      4. Simulate a stochastic reward (for demo purposes) and return the update.
    """
    actions = [BanditAction(aid) for aid in candidate_actions]
    refresh_actions(actions, r, delta)
    chosen = select_ucb_action(actions)

    # ----- Demo reward generation -------------------------------------------------
    # In a real system the reward would come from the environment.
    simulated_reward = random.random()  # uniform reward in [0,1)
    update = BanditUpdate(
        context_id=context_id,
        action_id=chosen.action_id,
        reward=simulated_reward,
        propensity=1.0 / len(candidate_actions),
    )
    # ------------------------------------------------------------------------------

    return chosen, [update]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Reset any lingering state
    reset_policy()

    # Simulate a stream of 50 contexts with three possible actions each
    actions_pool = ["codex", "groq", "cohere"]
    for t in range(50):
        ctx = f"ctx_{t}"
        chosen_action, updates = hybrid_bandit_step(ctx, actions_pool, r=1.0, delta=0.05)
        update_policy(updates)

    # After learning, display final UCB scores
    final_actions = [BanditAction(a) for a in actions_pool]
    refresh_actions(final_actions, r=1.0, delta=0.05)
    for a in final_actions:
        print(
            f"Action {a.action_id}: avg_reward={a.expected_reward:.4f}, "
            f"conf={a.confidence_bound:.4f}, ucb={a.ucb_score():.4f}"
        )

    # Demonstrate Count‑Min Sketch on the action ids observed in updates
    all_action_ids = [u.action_id for upd in _POLICY.values() for u in []]  # placeholder
    # For demo we just sketch the pool
    cms = count_min_sketch(actions_pool, width=32, depth=3)
    for aid in actions_pool:
        est = cms_estimate(cms, aid)
        print(f"CMS estimate for '{aid}': {est}")