# DARWIN HAMMER — match 20, survivor 2
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:23:03Z

"""Hybrid bandit-router and Schoolfield temperature model.

This module fuses two previously independent algorithms:

* **Bandit router (parent A)** – a lightweight multi‑armed bandit selector that
  computes a reward estimate `_reward(a)` and a confidence term based on the
  number of observations for each action.

* **Schoolfield poikilotherm model (parent B)** – a biologically‑inspired
  temperature‑dependent rate function `developmental_rate` that yields a
  normalized activity gate in the interval [0, 1].

**Mathematical bridge**

Both parents expose a scalar that can be interpreted as a *gain* applied to a
base quantity:

* In the bandit router the exploration term is `scale / sqrt(1 + n_a)`,
  where `scale = sqrt(∑ v_i²)` is the Euclidean norm of the context vector.
* In the Schoolfield model the normalized activity `A(T)` (0 ≤ A ≤ 1) is a
  smooth, temperature‑dependent scaling factor.

The hybrid algorithm multiplies the context norm by the activity gate,
producing a **temperature‑aware scale**  


    S_T = A(T) * √(∑_i context_i²)


and feeds `S_T` into the bandit’s upper‑confidence‑bound (UCB) term.  In this
way the exploration/exploitation balance is modulated by the operating
temperature: when the system is in its optimal thermal band (`A≈1`) the
bandit behaves normally; when temperature is too low or too high
(`A≈0`) the confidence term shrinks, biasing the selector toward actions with
higher empirical reward.

The module provides three core hybrid functions:

* `temperature_activity` – compute the normalized activity gate from Celsius.
* `hybrid_select_action` – temperature‑aware bandit action selection.
* `hybrid_update_policy` – update the policy with temperature‑weighted rewards.

All other utilities from the original parents (policy storage, reset, etc.)
are retained for compatibility.
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
# Parent A – bandit router core (lightly adapted)
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


# Global policy storage: action_id -> [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear all stored reward statistics."""
    _POLICY.clear()


def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Parent B – Schoolfield temperature model (lightly adapted)
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


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Full Schoolfield rate as a function of absolute temperature."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)


def normalized_activity(
    temp_c: float,
    low_c: float = 5.0,
    high_c: float = 40.0,
    samples: int = 141,
) -> float:
    """
    Map an observed operating temperature (°C) to a 0‥1 activity gate using
    the Schoolfield curve.  The gate is the rate divided by its maximal value
    over the specified temperature interval.
    """
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)

    # Pre‑compute the maximum rate over the interval [low_c, high_c]
    temps = np.linspace(low_c, high_c, max(2, samples))
    rates = np.vectorize(lambda t: developmental_rate(c_to_k(t), params))(temps)
    max_rate = np.max(rates)

    if max_rate <= 0:
        return 0.0
    return max(0.0, min(1.0, rate / max_rate))


# ----------------------------------------------------------------------
# Hybrid layer – temperature‑aware bandit selection
# ----------------------------------------------------------------------


def temperature_activity(temp_c: float) -> float:
    """
    Wrapper around ``normalized_activity`` that catches errors and guarantees a
    float in [0, 1].
    """
    try:
        return normalized_activity(temp_c)
    except Exception:
        return 0.0


def _temperature_scaled_scale(context: Dict[str, float], temp_c: float) -> float:
    """
    Compute the temperature‑scaled context norm:

        S_T = A(T) * √(∑ v_i²)

    where A(T) is the activity gate.
    """
    if not context:
        return 1.0
    vec = np.fromiter(context.values(), dtype=float)
    norm = float(np.linalg.norm(vec))
    activity = temperature_activity(temp_c)
    return activity * norm if activity > 0 else 1e-6  # avoid division by zero


def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    temperature_c: float,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a temperature‑aware bandit policy.

    Parameters
    ----------
    context: mapping of feature name → value (used for scaling)
    actions: list of candidate action identifiers
    temperature_c: current operating temperature in Celsius
    algorithm: one of ``'linucb'``, ``'epsilon_greedy'``, ``'thompson'``
    epsilon: exploration probability for epsilon‑greedy
    seed: random seed (int, str or None)

    Returns
    -------
    BanditAction – a frozen dataclass containing the chosen action and
    associated statistics.
    """
    if not actions:
        raise ValueError("actions required")

    rng = random.Random(seed)

    # ---- epsilon‑greedy -------------------------------------------------
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
        return BanditAction(
            action_id=chosen,
            propensity=1.0 / len(actions),
            expected_reward=_reward(chosen),
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    # ---- Thompson sampling ---------------------------------------------
    if algorithm == "thompson":
        # Beta posterior parameters: α = 1 + successes, β = 1 + failures
        # Use temperature‑scaled reward as “success” proxy.
        def thompson_score(a: str) -> float:
            mean = _reward(a)
            successes = max(0.0, mean) * temperature_activity(temperature_c) * 10.0
            failures = max(0.0, 1.0 - mean) * (1.0 - temperature_activity(temperature_c)) * 10.0
            return rng.betavariate(1 + successes, 1 + failures)

        chosen = max(actions, key=thompson_score)
        return BanditAction(
            action_id=chosen,
            propensity=1.0 / len(actions),
            expected_reward=_reward(chosen),
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    # ---- LinUCB / UCB variant -------------------------------------------
    # Temperature‑scaled context norm replaces the plain norm.
    scale = _temperature_scaled_scale(context, temperature_c)

    def ucb_score(a: str) -> float:
        mean = _reward(a)
        n = _POLICY.get(a, [0.0, 0.0])[1]
        confidence = 0.1 * scale / math.sqrt(1.0 + n)
        return mean + confidence

    chosen = max(actions, key=ucb_score)
    n_chosen = _POLICY.get(chosen, [0.0, 0.0])[1]
    confidence = 0.1 * scale / math.sqrt(1.0 + n_chosen)

    return BanditAction(
        action_id=chosen,
        propensity=1.0 / len(actions),
        expected_reward=_reward(chosen),
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def hybrid_update_policy(
    updates: List[BanditUpdate],
    temperature_c: float,
) -> None:
    """
    Update the global policy with temperature‑weighted rewards.

    Each reward is multiplied by the activity gate A(T) before being
    incorporated, ensuring that observations made under sub‑optimal
    temperatures have proportionally less influence on the learned averages.
    """
    activity = temperature_activity(temperature_c)
    for u in updates:
        weighted_reward = float(u.reward) * activity
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += weighted_reward
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------


def simulate_round(
    context: Dict[str, float],
    actions: List[str],
    true_rewards: Dict[str, float],
    temperature_c: float,
    algorithm: str = "linucb",
) -> Tuple[BanditAction, BanditUpdate]:
    """
    Simulate a single decision‑reward cycle:

    1. Select an action with ``hybrid_select_action``.
    2. Observe a stochastic reward drawn from a Bernoulli with mean
       ``true_rewards[action]``.
    3. Return both the selected action and a ``BanditUpdate`` record.
    """
    selected = hybrid_select_action(
        context=context,
        actions=actions,
        temperature_c=temperature_c,
        algorithm=algorithm,
    )
    # Stochastic reward (Bernoulli)
    rng = random.Random()
    reward = 1.0 if rng.random() < true_rewards.get(selected.action_id, 0.0) else 0.0
    update = BanditUpdate(
        context_id="sim",
        action_id=selected.action_id,
        reward=reward,
        propensity=selected.propensity,
    )
    return selected, update


def run_demo(num_rounds: int = 100) -> None:
    """Run a quick Monte‑Carlo demo of the hybrid bandit under varying temperature."""
    actions = ["alpha", "beta", "gamma"]
    # Ground‑truth reward probabilities (static)
    true_rewards = {"alpha": 0.6, "beta": 0.4, "gamma": 0.2}
    # Context features (arbitrary)
    base_context = {"cpu_load": 0.3, "mem_util": 0.5, "latency": 0.1}
    # Temperature schedule: start optimal, then drift cold, then hot
    temps = np.concatenate(
        [
            np.full(num_rounds // 3, 22.0),  # optimal ~22 °C
            np.linspace(22.0, 5.0, num_rounds // 3),  # cooling
            np.linspace(22.0, 45.0, num_rounds - 2 * (num_rounds // 3)),  # heating
        ]
    )
    for i in range(num_rounds):
        temp = float(temps[i])
        # Slightly perturb context each round
        context = {k: v + random.uniform(-0.02, 0.02) for k, v in base_context.items()}
        action, upd = simulate_round(
            context=context,
            actions=actions,
            true_rewards=true_rewards,
            temperature_c=temp,
            algorithm="linucb",
        )
        hybrid_update_policy([upd], temperature_c=temp)

    # Summarize learned averages
    print("Final estimated rewards (temperature‑aware):")
    for a in actions:
        print(f"  {a}: { _reward(a):.3f} (count={_POLICY.get(a, [0,0])[1]})")
    print(f"Final activity gate at 22 °C: {temperature_activity(22.0):.3f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Ensure deterministic behaviour for the demo
    random.seed(42)
    np.random.seed(0)

    reset_policy()
    print("Running hybrid bandit‑Schoolfield demo...")
    run_demo(num_rounds=120)
    print("Demo completed without error.")