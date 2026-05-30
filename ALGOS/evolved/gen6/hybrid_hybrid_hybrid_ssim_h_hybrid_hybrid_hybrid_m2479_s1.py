# DARWIN HAMMER — match 2479, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_ssim_hybrid_h_hybrid_sketches_rlct_m1064_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s4.py (gen3)
# born: 2026-05-29T23:42:38Z

"""Hybrid algorithm merging geometric‑multivector similarity (Parent A) with
a contextual multi‑armed bandit that uses a virtual store (Parent B).

Mathematical bridge
-------------------
* Each action and the current context are represented as 1‑D numeric
  sequences.
* `stats_to_multivector` builds a grade‑0/1/2 multivector from the
  first two statistical moments of a sequence.
* `geometric_ssim` returns the scalar part of the geometric product of the
  two multivectors – a similarity score `S(context, action)`.
* The similarity is fed as an **inflow** term to the virtual‑store dynamics
  of the bandit:


store_{t+1} = store_t + Δt·(α·S – β·store_t)


* The bandit’s expected reward for an action is the sum of the similarity
  and the current store value, i.e. `r̂ = S + store`.  Propensities are
  derived from a soft‑max over these expected rewards, while confidence
  bounds follow the Hoeffding inequality from Parent B.

The resulting system jointly exploits geometric‑algebraic similarity and
online bandit learning in a single, mathematically consistent loop.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Multivector and geometric similarity
# ----------------------------------------------------------------------


class Multivector:
    """Simple Euclidean Clifford algebra up to grade 2."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # Remove near‑zero components for cleanliness
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)  # dimension of the underlying vector space

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            result[blade] = result.get(blade, 0.0) + value
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (here reduced to simple outer product)."""
        result: Dict[FrozenSet[int], float] = {}
        for blade1, value1 in self.components.items():
            for blade2, value2 in other.components.items():
                new_blade = frozenset(blade1.union(blade2))
                result[new_blade] = result.get(new_blade, 0.0) + value1 * value2
        return Multivector(result, self.n)


def stats_to_multivector(seq: Sequence[float]) -> Multivector:
    """Convert a 1‑D sequence into a multivector of moments."""
    if len(seq) == 0:
        mean = 0.0
        var = 0.0
    else:
        mean = float(np.mean(seq))
        var = float(np.var(seq))
    # For demonstration we set a dummy covariance (grade‑2) to zero.
    components = {
        frozenset(): mean,               # scalar (grade‑0)
        frozenset([0]): var,            # grade‑1 placeholder
        frozenset([0, 1]): 0.0,         # grade‑2 placeholder
    }
    return Multivector(components, 2)


def geometric_ssim(x: Sequence[float], y: Sequence[float]) -> float:
    """Hybrid similarity using the geometric product of the moment multivectors."""
    mx = stats_to_multivector(x)
    my = stats_to_multivector(y)
    product = mx * my
    return product.scalar_part()


# ----------------------------------------------------------------------
# Parent B – Contextual bandit with virtual store
# ----------------------------------------------------------------------


# Shared constants (tuned for the hybrid)
ALPHA = 0.6          # store inflow coefficient (multiplied by similarity)
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for policy updates
DELTA_MAX = 1.0      # max evasion magnitude (unused in this hybrid)
ALPHA_EVASION = 3.0  # decay rate for evasion schedule (unused)
HOEFFDING_DELTA = 0.05  # confidence level for Hoeffding bound
CLAMP_LO = -5.0
CLAMP_HI = 5.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability of selection (influent)
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# Global mutable state (mirrors Parent B)
_POLICY: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}          # virtual store per action


def reset_policy() -> None:
    """Clear learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward_estimate(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def _confidence_bound(action_id: str) -> float:
    """Hoeffding‑type confidence bound for the given action."""
    _, n = _POLICY.get(action_id, [0.0, 0.0])
    if n == 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / HOEFFDING_DELTA) / (2.0 * n))


def _softmax(values: List[float]) -> List[float]:
    """Numerically stable soft‑max."""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def compute_similarity(context: Sequence[float], action_features: Sequence[float]) -> float:
    """Return geometric similarity S(context, action)."""
    return geometric_ssim(context, action_features)


def update_virtual_store(action_id: str, similarity: float) -> None:
    """
    Update the virtual store for *action_id* using the inflow
    proportional to the similarity score.
    """
    store = _STORE.get(action_id, 0.0)
    delta = DT * (ALPHA * similarity - BETA * store)
    _STORE[action_id] = store + delta


def select_hybrid_action(
    context: Sequence[float],
    actions_features: Dict[str, Sequence[float]],
    epsilon: float = 0.1,
    seed: int | None = None,
) -> BanditAction:
    """
    Choose an action using an ε‑greedy policy over soft‑max propensities.
    Expected reward for each action is `similarity + store`.
    """
    if seed is not None:
        random.seed(seed)

    if not actions_features:
        raise ValueError("actions_features cannot be empty")

    # Compute similarity and update stores
    sims: Dict[str, float] = {}
    for aid, feats in actions_features.items():
        s = compute_similarity(context, feats)
        sims[aid] = s
        update_virtual_store(aid, s)

    # Expected reward = similarity + current store value
    exp_rewards = {
        aid: sims[aid] + _STORE.get(aid, 0.0) for aid in actions_features
    }

    # Propensity via soft‑max over expected rewards
    reward_vals = list(exp_rewards.values())
    propensities = _softmax(reward_vals)
    action_ids = list(exp_rewards.keys())

    # ε‑greedy exploration
    if random.random() < epsilon:
        chosen_id = random.choice(action_ids)
    else:
        # Choose the action with highest propensity
        max_idx = max(range(len(propensities)), key=lambda i: propensities[i])
        chosen_id = action_ids[max_idx]

    chosen_propensity = propensities[action_ids.index(chosen_id)]
    chosen_reward = exp_rewards[chosen_id]
    confidence = _confidence_bound(chosen_id)

    return BanditAction(
        action_id=chosen_id,
        propensity=chosen_propensity,
        expected_reward=chosen_reward,
        confidence_bound=confidence,
    )


def bandit_update(action: BanditAction, reward: float, propensity: float) -> None:
    """
    Record the observed reward and update policy statistics.
    """
    total, count = _POLICY.get(action.action_id, [0.0, 0.0])
    total += reward
    count += 1
    _POLICY[action.action_id] = [total, count]


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo() -> None:
    """Run a short simulation to verify that the hybrid loop executes."""
    reset_policy()

    # Synthetic context and three actions with random feature vectors
    context = np.random.randn(10).tolist()
    actions_features = {
        "A": np.random.randn(10).tolist(),
        "B": np.random.randn(10).tolist(),
        "C": np.random.randn(10).tolist(),
    }

    for step in range(5):
        # Select action
        chosen = select_hybrid_action(context, actions_features, epsilon=0.2, seed=step)
        # Simulate a stochastic reward (here simply the similarity plus noise)
        true_similarity = compute_similarity(context, actions_features[chosen.action_id])
        noise = random.gauss(0, 0.1)
        observed_reward = true_similarity + noise

        # Update bandit statistics
        bandit_update(chosen, observed_reward, chosen.propensity)

        # Print a concise summary
        print(
            f"Step {step}: chosen={chosen.action_id}, "
            f"exp_reward={chosen.expected_reward:.4f}, "
            f"obs_reward={observed_reward:.4f}, "
            f"store={_STORE[chosen.action_id]:.4f}"
        )

    # Final policy snapshot
    print("\nFinal policy statistics:")
    for aid, (tot, cnt) in _POLICY.items():
        print(f"  {aid}: total={tot:.4f}, count={cnt}, avg={tot/cnt if cnt else 0:.4f}")


if __name__ == "__main__":
    _demo()