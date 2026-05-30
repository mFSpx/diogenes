# DARWIN HAMMER — match 2771, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_indy_learning_hybrid_hybrid_distri_m1222_s8.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2377_s0.py (gen6)
# born: 2026-05-29T23:45:45Z

"""Hybrid Bandit‑Perceptual‑Fractional Fusion (Parents A & B)

This module combines the contextual bandit core from *Parent A* with the
perceptual‑hash, Fisher‑localization and Caputo‑fractional utilities from
*Parent B*.  

Mathematical bridge  
-------------------
1. **Health‑score vector** – the raw context vector is treated as a health‑score
   vector *h*.  
2. **Expected reward** – the bandit’s expected reward for an arm *a* is the
   empirical average reward stored in the global policy (Parent A).  
3. **Regret** – for a given context the regret is  
   ``r = max_i h_i – h_a`` (max‑plus tropical algebra).  
4. **Caputo derivative** – a discrete Grünwald‑Letnikov approximation of the
   Caputo fractional derivative of the regret history of each arm is used as
   the *propensity* score that drives exploration (Parent B).  
5. **Perceptual hash** – the first 64 bits of the context are hashed to obtain a
   deterministic arm identifier, providing a fast lookup into the bandit store
   (Parent B).  
6. **Fisher intensity** – the variance of the context vector approximates a
   Fisher‑information‑like intensity, which modulates a work‑share allocation
   after the bandit decision (Parent B).

The fusion yields a single unified system where a context is hashed → an arm
is selected with a UCB‑style confidence bound, its propensity is shaped by the
fractional derivative of past regret, and the resulting decision is weighted
by a Fisher‑derived intensity.  
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Global stores (Parent A)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action_id -> [cum_reward, count, last_propensity]
_REGRET_SERIES: Dict[str, List[float]] = {}  # action_id -> list of past regrets


def reset_policy() -> None:
    """Clear all stored statistics."""
    _POLICY.clear()
    _REGRET_SERIES.clear()


def _reward(action: str) -> float:
    total, n, _ = _POLICY.get(action, [0.0, 0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0, 0.0])[1]


def update_policy(updates: List["BanditUpdate"]) -> None:
    """Batch update of the global bandit policy."""
    for u in updates:
        total, n, _ = _POLICY.get(u.action_id, [0.0, 0.0, 0.0])
        _POLICY[u.action_id] = [total + u.reward, n + 1, u.propensity]
        # keep regret history for the fractional derivative
        regret = u.reward - _reward(u.action_id)  # signed regret of this step
        _REGRET_SERIES.setdefault(u.action_id, []).append(regret)


# ----------------------------------------------------------------------
# Perceptual hash (Parent B)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """
    64‑bit perceptual hash based on average comparison.
    Returns an integer in [0, 2**64).
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    # pad remaining bits with zeros if less than 64 values
    bits <<= max(0, 64 - len(values))
    return bits


# ----------------------------------------------------------------------
# Caputo fractional derivative (Parent B)
# ----------------------------------------------------------------------
def _binomial(alpha: float, k: int) -> float:
    """Generalised binomial coefficient (α choose k)."""
    return math.gamma(alpha + 1) / (math.gamma(k + 1) * math.gamma(alpha - k + 1))


def caputo_fractional_derivative(series: List[float], alpha: float = 0.5) -> float:
    """
    Discrete Grünwald‑Letnikov approximation of the Caputo derivative
    D^α f(t) at the last point of *series* (step size h = 1).
    """
    n = len(series) - 1
    if n <= 0:
        return 0.0
    coeff_sum = 0.0
    for k in range(n + 1):
        coeff = (-1) ** k * _binomial(alpha, k)
        coeff_sum += coeff * series[n - k]
    return coeff_sum / (1 ** alpha)  # h = 1


# ----------------------------------------------------------------------
# Tropical Max‑Plus (Parent B)
# ----------------------------------------------------------------------
def tropical_max_plus(health: List[float], action_idx: int) -> float:
    """
    Tropical max‑plus regret for a given action.
    Regret = max_i (health_i) - (health_action_idx)
    """
    if not health:
        return 0.0
    max_h = max(health)
    return max_h - health[action_idx]


# ----------------------------------------------------------------------
# Fisher intensity (Parent B)
# ----------------------------------------------------------------------
def fisher_intensity(values: List[float]) -> float:
    """
    Simple proxy for Fisher information: variance scaled by vector length.
    """
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    var = np.var(arr)
    return var * len(arr)


# ----------------------------------------------------------------------
# Data structures (unified)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid_fusion"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def select_action(context_id: str, values: List[float]) -> BanditAction:
    """
    Hash the context to an arm, compute expected reward, UCB confidence,
    and a propensity derived from the Caputo fractional derivative of the
    regret history.
    """
    # 1. Perceptual hash → deterministic arm identifier
    action_hash = compute_phash(values)
    action_id = f"arm_{action_hash:x}"

    # 2. Expected reward from policy
    exp_reward = _reward(action_id)

    # 3. UCB‑style confidence bound
    total_counts = sum(_count(a) for a in _POLICY) + 1.0
    n_a = _count(action_id) + 1e-6  # avoid division by zero
    confidence = math.sqrt(2 * math.log(total_counts) / n_a)

    # 4. Regret (tropical max‑plus) and its fractional derivative
    #    Use the index of the hashed arm modulo length of the vector as a proxy.
    idx = action_hash % max(1, len(values))
    regret = tropical_max_plus(values, idx)
    _REGRET_SERIES.setdefault(action_id, []).append(regret)
    propensity_raw = caputo_fractional_derivative(_REGRET_SERIES[action_id], alpha=0.5)

    # Normalize propensity to [0,1] using a sigmoid
    propensity = 1.0 / (1.0 + math.exp(-propensity_raw))

    return BanditAction(
        action_id=action_id,
        propensity=propensity,
        expected_reward=exp_reward,
        confidence_bound=confidence,
    )


def process_feedback(context_id: str, action: BanditAction, reward: float) -> None:
    """
    Record the outcome of an arm pull and update the global policy.
    """
    update = BanditUpdate(
        context_id=context_id,
        action_id=action.action_id,
        reward=reward,
        propensity=action.propensity,
    )
    update_policy([update])


def allocate_workshare(action: BanditAction, values: List[float]) -> float:
    """
    Compute a work‑share allocation proportional to the Fisher intensity
    of the context and the action's propensity.
    """
    intensity = fisher_intensity(values)
    allocation = intensity * action.propensity
    # Clip to a reasonable range
    return max(0.0, min(allocation, 1.0))


# ----------------------------------------------------------------------
# Example StoreState (illustrative, not used directly in the test)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Simple linear reservoir update:
        Δlevel = α·Σ(inflow) – β·Σ(outflow)
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, min(self.level + delta * self.dt, self.limit))
        return self.level, delta


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    reset_policy()

    # Simulated hidden weight vector for generating rewards
    hidden_weight = np.random.randn(10)

    for step in range(5):
        ctx_id = f"ctx_{step}"
        # Random context of length 10
        ctx = np.random.randn(10).tolist()

        # Bandit decision
        act = select_action(ctx_id, ctx)

        # Simulated reward: dot product + Gaussian noise
        reward = float(np.dot(ctx, hidden_weight) + random.gauss(0, 0.1))

        # Record feedback
        process_feedback(ctx_id, act, reward)

        # Work‑share allocation demonstration
        alloc = allocate_workshare(act, ctx)

        print(
            f"Step {step}: action={act.action_id[:8]}..., "
            f"exp_reward={act.expected_reward:.3f}, "
            f"propensity={act.propensity:.3f}, "
            f"confidence={act.confidence_bound:.3f}, "
            f"reward={reward:.3f}, "
            f"allocation={alloc:.3f}"
        )