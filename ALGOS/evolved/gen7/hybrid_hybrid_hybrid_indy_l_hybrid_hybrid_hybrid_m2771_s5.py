# DARWIN HAMMER — match 2771, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_indy_learning_hybrid_hybrid_distri_m1222_s8.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2377_s0.py (gen6)
# born: 2026-05-29T23:45:45Z

"""Hybrid Bandit‑Store Fusion (Parents A & B)

This module merges the contextual bandit core from *Parent A* with the
state‑store dynamics and fractional‑Caputo utilities from *Parent B*.

Mathematical bridge
-------------------
- The **expected reward** of each `BanditAction` (Parent A) is taken as the
  health‑score vector produced by the `StoreState` dynamics (Parent B).
- The **propensity** for an action is generated from a Caputo fractional
  derivative of the regret‑adjusted gain; this couples the tropical‑max‑plus
  intuition of Parent B to the bandit exploration term.
- A **Fisher‑based intensity** derived from the context vector modulates the
  work‑share allocation, completing the fusion of the two lineages.

The resulting system can be used as a lightweight reinforcement‑learning
router that respects both cumulative reward tracking and fractional‑order
adaptivity.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared Bandit structures (derived from Parent A & B)
# ----------------------------------------------------------------------
class BanditAction:
    """Container for an action in the contextual bandit."""
    def __init__(
        self,
        action_id: str,
        expected_reward: float = 0.0,
        propensity: float = 0.0,
        confidence_bound: float = 0.0,
        algorithm: str = "hybrid",
    ):
        self.action_id = action_id
        self.expected_reward = expected_reward
        self.propensity = propensity
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

    def __repr__(self) -> str:
        return (
            f"BanditAction(id={self.action_id}, exp={self.expected_reward:.3f}, "
            f"prop={self.propensity:.3f}, bound={self.confidence_bound:.3f})"
        )


class BanditUpdate:
    """Result of pulling an arm."""
    def __init__(
        self,
        context_id: str,
        action_id: str,
        reward: float,
        propensity: float,
    ):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity


# Global policy store (Parent A)
_POLICY: Dict[str, List[float]] = {}  # action_id -> [cumulative_reward, count, last_propensity]


def reset_policy() -> None:
    """Erase all stored statistics."""
    _POLICY.clear()


def _reward(action: str) -> float:
    total, n, _ = _POLICY.get(action, [0.0, 0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """Batch‑apply bandit updates to the global policy."""
    for u in updates:
        total, n, _ = _POLICY.get(u.action_id, [0.0, 0.0, 0.0])
        _POLICY[u.action_id] = [total + u.reward, n + 1, u.propensity]


# ----------------------------------------------------------------------
# Perceptual hash (Parent A fragment, fixed)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash based on average comparison."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


# ----------------------------------------------------------------------
# Fractional Caputo derivative (Parent B inspiration)
# ----------------------------------------------------------------------
def caputo_derivative(signal: np.ndarray, order: float, dt: float = 1.0) -> np.ndarray:
    """
    Approximate the Caputo fractional derivative of ``signal`` of order ``order``.
    Uses the Grünwald‑Letnikov scheme with a simple truncation.

    Parameters
    ----------
    signal : np.ndarray
        1‑D array of sampled values.
    order : float
        Fractional order 0 < order < 1.
    dt : float
        Sampling interval.

    Returns
    -------
    np.ndarray
        Approximation of the derivative, same length as ``signal`` (first
        ``k`` entries are zero where the stencil is incomplete).
    """
    if not (0.0 < order < 1.0):
        raise ValueError("order must be in (0,1) for this implementation")
    n = len(signal)
    derivative = np.zeros_like(signal, dtype=float)

    # Pre‑compute binomial coefficients
    coeffs = np.zeros(n)
    coeffs[0] = 1.0
    for k in range(1, n):
        coeffs[k] = coeffs[k - 1] * (order - k + 1) / k

    for i in range(1, n):
        k_max = i
        derivative[i] = np.sum(coeffs[:k_max + 1][::-1] * signal[: k_max + 1])
        derivative[i] /= dt ** order

    return derivative


# ----------------------------------------------------------------------
# StoreState (Parent B fragment, completed)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    """Dynamic store used to compute health‑score vectors."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Perform a tropical‑max‑plus style update.

        Returns
        -------
        Tuple[float, float]
            New level and updated gain.
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        # Tropical max‑plus: level = max(base, level + delta)
        self.level = max(self.base, self.level + delta)

        # Apply a simple Caputo‑adjusted gain modulation
        recent = np.array([self.level, delta])
        caputo_adj = caputo_derivative(recent, order=0.5, dt=self.dt)[-1]
        self.gain = max(0.0, self.gain + caputo_adj)

        # Clip to limits
        self.level = min(self.level, self.limit)
        self.gain = min(self.gain, self.limit)

        return self.level, self.gain


# ----------------------------------------------------------------------
# Fisher‑based intensity (Parent B inspiration)
# ----------------------------------------------------------------------
def fisher_intensity(values: List[float]) -> float:
    """
    Compute a scalar intensity from ``values`` using a proxy for Fisher
    information: the inverse of the sample variance (with a small epsilon).

    This intensity modulates the work‑share allocation.
    """
    if len(values) < 2:
        return 0.0
    var = np.var(values, ddof=1)
    eps = 1e-9
    return 1.0 / (var + eps)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_compute_expected_rewards(state: StoreState, actions: List[str]) -> Dict[str, float]:
    """
    Use the current ``StoreState`` level as a health‑score vector and map it
    to expected rewards for each action via a simple affine transform.
    """
    base_reward = math.tanh(state.level)  # bounded between -1 and 1
    rewards = {}
    for i, a in enumerate(actions):
        # Small perturbation per action to break ties
        rewards[a] = base_reward + 0.01 * math.sin(i)
    return rewards


def hybrid_select_action(
    state: StoreState,
    context_vec: List[float],
    action_ids: List[str],
) -> BanditAction:
    """
    Choose an action by marrying bandit statistics with fractional‑order
    propensity computation.

    Steps
    -----
    1. Compute expected rewards from the store (health scores).
    2. Retrieve empirical rewards from the global policy.
    3. Form a regret term = max_empirical - empirical.
    4. Apply a Caputo derivative to the regret vector → propensity.
    5. Return the action maximizing ``expected_reward * propensity``.
    """
    # 1. Expected rewards from store
    exp_rewards = hybrid_compute_expected_rewards(state, action_ids)

    # 2. Empirical rewards from policy
    empirical = {a: _reward(a) for a in action_ids}
    max_emp = max(empirical.values()) if empirical else 0.0

    # 3. Regret vector
    regret = np.array([max_emp - empirical.get(a, 0.0) for a in action_ids])

    # 4. Fractional propensity
    prop_raw = caputo_derivative(regret, order=0.5, dt=state.dt)
    # Normalise to [0,1] via sigmoid
    prop = 1.0 / (1.0 + np.exp(-prop_raw))

    # 5. Score and select
    best_score = -math.inf
    best_action = None
    for i, aid in enumerate(action_ids):
        score = exp_rewards[aid] * prop[i]
        if score > best_score:
            best_score = score
            best_action = BanditAction(
                action_id=aid,
                expected_reward=exp_rewards[aid],
                propensity=prop[i],
                confidence_bound=0.0,  # placeholder
                algorithm="hybrid",
            )
    return best_action


def hybrid_step(
    state: StoreState,
    context_id: str,
    context_vec: List[float],
    action_ids: List[str],
) -> Tuple[BanditUpdate, StoreState]:
    """
    Perform a single hybrid iteration:
    * select an action,
    * compute a reward (synthetic for demo purposes),
    * update the global bandit policy,
    * evolve the store state using inflow/outflow derived from the reward.
    """
    # Select action
    chosen = hybrid_select_action(state, context_vec, action_ids)

    # Synthetic reward: dot product of context with a random weight vector,
    # scaled by the chosen propensity.
    weight = np.random.RandomState(hash(chosen.action_id) % (2 ** 32)).randn(len(context_vec))
    raw_reward = float(np.dot(context_vec, weight))
    reward = chosen.propensity * math.tanh(raw_reward)  # bound reward

    # Create update record
    update = BanditUpdate(
        context_id=context_id,
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )

    # Apply policy update
    update_policy([update])

    # Compute inflow/outflow for StoreState
    intensity = fisher_intensity(context_vec)
    inflow = [reward * intensity]
    outflow = [chosen.propensity * intensity]

    # Evolve store
    state.update(inflow, outflow)

    return update, state


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise
    random.seed(42)
    np.random.seed(42)

    store = StoreState(level=1.0, alpha=0.7, beta=0.3, dt=1.0, base=0.0, limit=5.0)

    actions = [f"arm_{i}" for i in range(5)]
    context_vector = [random.random() for _ in range(8)]

    # Run a few hybrid steps
    for step in range(10):
        upd, store = hybrid_step(
            state=store,
            context_id=f"ctx_{step}",
            context_vec=context_vector,
            action_ids=actions,
        )
        print(
            f"Step {step:02d}: selected {upd.action_id}, reward={upd.reward:.3f}, "
            f"propensity={upd.propensity:.3f}, level={store.level:.3f}, gain={store.gain:.3f}"
        )

    # Demonstrate perceptual hash utility
    h = compute_phash(context_vector)
    print(f"Perceptual hash of last context vector: {h:#018x}")