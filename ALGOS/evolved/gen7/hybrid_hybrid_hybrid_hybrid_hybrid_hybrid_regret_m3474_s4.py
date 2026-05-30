# DARWIN HAMMER — match 3474, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1715_s0.py (gen6)
# parent_b: hybrid_hybrid_regret_engine_hybrid_decision_hygi_m993_s2.py (gen3)
# born: 2026-05-29T23:50:19Z

"""Hybrid Algorithm integrating Bandit confidence modulation with Regret‑Entropy dynamics.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1715_s0.py (Bandit with allocation routine, confidence bounds, and Koopman linearisation)
- hybrid_hybrid_regret_engine_hybrid_decision_hygi_m993_s2.py (Regret engine with Shannon entropy over regret‑weighted action distribution)

Mathematical Bridge:
The allocation weight vector derived from the weekday‑group mapping (Parent A) is used to scale the classic
UCB‑type confidence term. The regret for each action (max_expected – expected) yields a regret‑weighted probability
distribution; its Shannon entropy (Parent B) quantifies uncertainty. The entropy term multiplies the scaled confidence,
producing a single scalar score:

    score_i = μ_i +  β · (c_i · w_i) · (1 + H),

where μ_i is the expected reward, c_i the raw UCB confidence, w_i the allocation weight, β a tunable
exploration factor, and H the Shannon entropy of the regret‑weighted distribution. This couples the two
topologies into a unified decision‑making system.
"""

import sys
import math
import random
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Data structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """Unified action representation."""
    id: str
    expected_reward: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class Observation:
    """Observation used to update the policy (bandit update)."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places (consistent with Parent A)."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return a 0‑based weekday index (consistent with Parent A)."""
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    year -= month < 3
    return (year + year // 4 - year // 100 + year // 400 + t[month - 1] + day) % 7


def weekday_weight_vector(groups: Tuple[str, ...]) -> np.ndarray:
    """
    Build a weight vector based on the current weekday.
    Each group receives the same weight = (weekday+1)/7, providing a simple
    allocation routine that can later modulate confidence terms.
    """
    today = dt.date.today()
    wk = doomsday(today.year, today.month, today.day)  # 0=Sunday … 6=Saturday
    base_weight = (wk + 1) / 7.0
    return np.full(len(groups), base_weight, dtype=float)


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution (base‑e)."""
    # Guard against zero probabilities
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return -float(np.sum(probs * np.log(probs)))


def regret_weighted_distribution(actions: Sequence[Action]) -> np.ndarray:
    """
    Produce a probability distribution proportional to regret:
    r_i = max_j μ_j – μ_i   (non‑negative)
    p_i = r_i / Σ r_j   (if Σ r_j = 0 → uniform)
    """
    expected = np.array([a.expected_reward for a in actions], dtype=float)
    max_mu = np.max(expected)
    regrets = max_mu - expected
    total = regrets.sum()
    if total == 0.0:
        # all actions equal; return uniform distribution
        return np.full_like(regrets, 1.0 / len(regrets))
    return regrets / total


# ----------------------------------------------------------------------
# Hybrid operations (at least three functions)
# ----------------------------------------------------------------------
def compute_scaled_confidence(
    action_counts: np.ndarray,
    total_selections: int,
    allocation_weights: np.ndarray,
    exploration_coef: float = 1.0,
) -> np.ndarray:
    """
    Classic UCB confidence term scaled by allocation weights.

    c_i = sqrt( (2 * log(total)) / n_i )
    scaled_c_i = c_i * w_i
    """
    # Avoid division by zero
    safe_counts = np.where(action_counts == 0, 1, action_counts)
    raw_conf = np.sqrt((2.0 * math.log(max(total_selections, 1))) / safe_counts)
    scaled_conf = raw_conf * allocation_weights
    return scaled_conf


def compute_action_scores(
    actions: Sequence[Action],
    action_counts: np.ndarray,
    total_selections: int,
    allocation_weights: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Unified scoring function:

        score_i = μ_i + β * (scaled_conf_i) * (1 + H)

    where H is the Shannon entropy of the regret‑weighted distribution.
    """
    # 1. Scaled confidence
    scaled_conf = compute_scaled_confidence(
        action_counts, total_selections, allocation_weights, exploration_coef=beta
    )

    # 2. Entropy of regret‑weighted distribution
    probs = regret_weighted_distribution(actions)
    H = shannon_entropy(probs)

    # 3. Combine
    expected = np.array([a.expected_reward for a in actions], dtype=float)
    scores = expected + beta * scaled_conf * (1.0 + H)
    return scores


def select_action(
    context_id: str,
    actions: Sequence[Action],
    policy: Dict[str, np.ndarray],
    groups: Tuple[str, ...] = GROUPS,
    beta: float = 1.0,
) -> Tuple[Action, float]:
    """
    Choose an action for a given context using the hybrid scoring rule.
    Returns the selected Action and its propensity (softmax over scores).
    """
    # Retrieve or initialise counts for this context
    if context_id not in policy:
        policy[context_id] = np.zeros(len(actions), dtype=int)

    counts = policy[context_id]
    total = int(counts.sum())

    # Allocation weights derived from weekday‑group mapping
    allocation_weights = weekday_weight_vector(groups)
    # If there are more actions than groups, repeat the weight vector
    if len(actions) > len(allocation_weights):
        repeats = math.ceil(len(actions) / len(allocation_weights))
        allocation_weights = np.tile(allocation_weights, repeats)[: len(actions)]

    # Compute hybrid scores
    scores = compute_action_scores(
        actions, counts, total, allocation_weights, beta=beta
    )

    # Softmax to obtain a proper propensity distribution
    max_score = np.max(scores)  # for numerical stability
    exp_scores = np.exp(scores - max_score)
    propensity = exp_scores / exp_scores.sum()

    # Sample according to propensity
    chosen_idx = np.random.choice(len(actions), p=propensity)
    chosen_action = actions[chosen_idx]
    chosen_propensity = float(propensity[chosen_idx])

    # Update counts for the selected action (increment now; final update will use reward)
    counts[chosen_idx] += 1
    policy[context_id] = counts

    return chosen_action, chosen_propensity


def update_policy(
    policy: Dict[str, np.ndarray],
    observation: Observation,
    action_index_map: Dict[str, int],
) -> None:
    """
    Incorporate a reward observation into the policy.
    For simplicity, we maintain only selection counts; a full bandit would also
    maintain reward estimates. Here we adjust expected reward estimates stored
    externally (e.g., in a model) – the function demonstrates the update hook.
    """
    counts = policy.get(observation.context_id)
    if counts is None:
        # Should not happen; initialise lazily
        counts = np.zeros(len(action_index_map), dtype=int)
        policy[observation.context_id] = counts

    # No direct count change here because the count was already increased during selection.
    # In a richer implementation we would update reward estimates; omitted for brevity.
    # Placeholder to illustrate the update pathway.
    _ = observation  # suppress unused variable warning
    _ = action_index_map


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small set of dummy actions
    dummy_actions = [
        Action(id="a1", expected_reward=0.2),
        Action(id="a2", expected_reward=0.5),
        Action(id="a3", expected_reward=0.1),
        Action(id="a4", expected_reward=0.4),
    ]

    # Mapping from action id to index for update hook
    idx_map = {a.id: i for i, a in enumerate(dummy_actions)}

    # Policy dictionary: context_id -> selection counts
    policy_store: Dict[str, np.ndarray] = {}

    # Simulate a few rounds
    for step in range(10):
        ctx = "session_1"
        act, prop = select_action(ctx, dummy_actions, policy_store, beta=0.8)
        # Simulate a stochastic reward (Bernoulli with p = expected_reward)
        reward = 1.0 if random.random() < act.expected_reward else 0.0
        obs = Observation(
            context_id=ctx,
            action_id=act.id,
            reward=reward,
            propensity=prop,
        )
        update_policy(policy_store, obs, idx_map)

    # Print final counts for inspection
    final_counts = policy_store.get("session_1")
    print("Final selection counts per action:")
    for a, cnt in zip(dummy_actions, final_counts):
        print(f"  {a.id}: {cnt}")

    sys.exit(0)