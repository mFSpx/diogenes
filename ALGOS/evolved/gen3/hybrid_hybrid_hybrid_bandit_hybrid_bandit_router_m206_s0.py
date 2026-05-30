# DARWIN HAMMER — match 206, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3.py (gen2)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s5.py (gen1)
# born: 2026-05-29T23:27:44Z

"""Hybrid Bandit‑Schoolfield Model
================================

This module fuses two previously independent algorithms:

* **Bandit router (Parent A & B)** – a contextual multi‑armed bandit that
  maintains a global policy ``_POLICY`` mapping ``action_id`` → ``[cumulative_reward,
  count]`` and selects actions with an ``linucb``‑style surrogate.
* **Schoolfield temperature model (Parent B)** – a mechanistic description of
  temperature‑dependent biological rates.

The mathematical bridge is the *expected reward* of a bandit arm.  
Instead of a raw scalar, the expected reward is now the **developmental rate**
computed by the Schoolfield equation for a temperature associated with that arm.
Thus each action corresponds to a temperature regime; the bandit uses the
temperature‑driven rate as its reward signal while still preserving its
exploration‑exploitation machinery.

The resulting hybrid can be used, for example, to adaptively select the
optimal incubation temperature in a biological experiment while learning from
observed outcomes.

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
# Shared data structures (Bandit core)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


# Global policy storage: action_id -> [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear all stored reward statistics."""
    _POLICY.clear()


def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def _count(a: str) -> float:
    """Number of times action *a* has been observed."""
    return _POLICY.get(a, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Schoolfield temperature model (Parent B)
# ----------------------------------------------------------------------


R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15  # reference temperature (25 °C) in Kelvin


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def schoolfield_rate(
    temp_k: float, params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Compute the developmental rate at absolute temperature ``temp_k`` using the
    modified Schoolfield equation.

    The formulation follows:

        rate = rho_25 *
               exp( -ΔH_a / (R·T) ) /
               ( 1 + exp( ΔH_l / R * (1/T_l - 1/T) )
                     + exp( ΔH_h / R * (1/T - 1/T_h) ) )

    where the symbols correspond to the fields of ``SchoolfieldParams``.
    """
    R = params.r_cal
    T = temp_k

    # Activation term
    act = math.exp(-params.delta_h_activation / (R * T))

    # Low‑temperature deactivation term
    low = math.exp(
        params.delta_h_low / R * (1.0 / params.t_low - 1.0 / T)
    )

    # High‑temperature deactivation term
    high = math.exp(
        params.delta_h_high / R * (1.0 / T - 1.0 / params.t_high)
    )

    denom = 1.0 + low + high
    rate = params.rho_25 * act / denom
    return rate


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def compute_expected_rewards(
    actions: List[str],
    temps_c: List[float],
    params: SchoolfieldParams = SchoolfieldParams(),
) -> Dict[str, float]:
    """
    Map each ``action_id`` to its temperature‑driven expected reward.

    ``actions`` and ``temps_c`` must be of equal length; the i‑th action is
    associated with the i‑th temperature in Celsius.
    """
    if len(actions) != len(temps_c):
        raise ValueError("actions and temps_c must have the same length")
    rewards: Dict[str, float] = {}
    for act, t_c in zip(actions, temps_c):
        t_k = c_to_k(t_c)
        rewards[act] = schoolfield_rate(t_k, params)
    return rewards


def select_action(
    context: Dict[str, float],
    actions: List[str],
    temps_c: List[float],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a bandit policy whose expected reward is the
    temperature‑dependent developmental rate.

    The function first computes the *model‑based* expected reward for each
    action (via ``schoolfield_rate``) and then adds the LinUCB exploration term.
    """
    if not actions:
        raise ValueError("actions required")
    if len(actions) != len(temps_c):
        raise ValueError("actions and temps_c must be the same length")

    rng = random.Random(seed)

    # Model‑based expected reward (deterministic part)
    model_rewards = compute_expected_rewards(actions, temps_c)

    # Exploration term (same as original LinUCB surrogate)
    scale = math.sqrt(
        sum(float(v) * float(v) for v in context.values())
    ) if context else 1.0

    def score(a: str) -> float:
        # empirical reward from history + exploration bonus
        empirical = _reward(a)
        exploration = 0.1 * scale / math.sqrt(1 + _count(a))
        return empirical + exploration + model_rewards.get(a, 0.0)

    # Epsilon‑greedy fallback
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    else:
        chosen = max(actions, key=score)

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _count(chosen))
    expected = model_rewards[chosen]

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected,
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    temps_c: List[float],
    true_reward_fn,
    algorithm: str = "linucb",
    seed: int | str | None = 7,
) -> Tuple[BanditAction, BanditUpdate]:
    """
    Perform a single decision‑observation cycle:

    1. Select an action with ``select_action``.
    2. Obtain a stochastic reward from ``true_reward_fn`` (which may add
       measurement noise to the deterministic Schoolfield rate).
    3. Update the global policy with the observed reward.

    Returns the chosen ``BanditAction`` and the ``BanditUpdate`` that was
    applied.
    """
    ba = select_action(context, actions, temps_c, algorithm=algorithm, seed=seed)
    # The true reward is a callable that receives (action_id, temperature_C)
    reward = true_reward_fn(ba.action_id, dict(zip(actions, temps_c))[ba.action_id])
    upd = BanditUpdate(
        context_id=str(context),
        action_id=ba.action_id,
        reward=reward,
        propensity=ba.propensity,
    )
    update_policy([upd])
    return ba, upd


# ----------------------------------------------------------------------
# Simple stochastic reward generator for testing
# ----------------------------------------------------------------------


def noisy_schoolfield_reward(
    action_id: str, temp_c: float, noise_sd: float = 0.05
) -> float:
    """
    Return the deterministic Schoolfield rate for ``temp_c`` perturbed by
    Gaussian noise with standard deviation ``noise_sd``.
    """
    deterministic = schoolfield_rate(c_to_k(temp_c))
    noise = np.random.normal(0.0, noise_sd)
    return max(deterministic + noise, 0.0)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define three temperature regimes (in Celsius) and corresponding actions
    actions = ["cold", "optimal", "hot"]
    temps_c = [15.0, 25.0, 35.0]  # °C

    # Simple context: could be any numeric features; we use a dummy one
    context = {"dummy_feature": 1.0}

    # Run a few hybrid steps to demonstrate learning
    for step in range(10):
        ba, upd = hybrid_step(
            context=context,
            actions=actions,
            temps_c=temps_c,
            true_reward_fn=noisy_schoolfield_reward,
            algorithm="linucb",
            seed=step,
        )
        print(
            f"Step {step+1:02d}: chose '{ba.action_id}' (temp={temps_c[actions.index(ba.action_id)]}°C) "
            f"=> reward={upd.reward:.4f}, mean_reward={_reward(ba.action_id):.4f}"
        )

    # Show final policy statistics
    print("\nFinal policy statistics:")
    for a in actions:
        total, cnt = _POLICY.get(a, [0.0, 0.0])
        print(f"  {a}: count={int(cnt)}, mean_reward={_reward(a):.4f}")