# DARWIN HAMMER — match 20, survivor 5
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:23:03Z

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – bandit router core (lightly adapted)
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


@lru_cache(maxsize=32)
def _max_rate_over_interval(
    low_c: float,
    high_c: float,
    samples: int,
    params: SchoolfieldParams,
) -> float:
    """Pre‑compute the maximal Schoolfield rate on a temperature interval."""
    temps = np.linspace(low_c, high_c, max(2, samples))
    rates = np.vectorize(lambda t: developmental_rate(c_to_k(t), params))(temps)
    return float(np.max(rates))


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
    max_rate = _max_rate_over_interval(low_c, high_c, samples, params)

    if max_rate <= 0:
        return 0.0
    return max(0.0, min(1.0, rate / max_rate))


# ----------------------------------------------------------------------
# Hybrid layer – temperature‑aware bandit selection
# ----------------------------------------------------------------------


def temperature_activity(temp_c: float) -> float:
    """
    Safe wrapper around ``normalized_activity`` that guarantees a float in [0, 1].
    """
    try:
        return normalized_activity(temp_c)
    except Exception:
        return 0.0


def _temperature_scaled_scale(context: Dict[str, float], temp_c: float) -> float:
    """
    Compute the temperature‑scaled context norm:

        S_T = A(T) * √(∑ v_i²)

    where A(T) is the activity gate.  A tiny epsilon is returned when the
    activity gate is zero to keep the exploration term well‑behaved.
    """
    if not context:
        # No features → fall back to unit scale.
        return 1.0
    vec = np.fromiter(context.values(), dtype=float)
    norm = float(np.linalg.norm(vec))
    activity = temperature_activity(temp_c)
    return activity * norm if activity > 0 else 1e-6


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

    # ------------------------------------------------------------------
    # epsilon‑greedy
    # ------------------------------------------------------------------
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
        return BanditAction(
            action_id=chosen,
            propensity=1.0 / len(actions),
            expected_reward=_reward(chosen),
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    # ------------------------------------------------------------------
    # Thompson sampling
    # ------------------------------------------------------------------
    if algorithm == "thompson":
        # Use a Beta posterior with pseudo‑counts derived from the
        # temperature‑scaled empirical mean.
        def thompson_score(a: str) -> float:
            mean = _reward(a)
            # Scale the mean into a pseudo‑count; the factor 10.0 is
            # arbitrary but provides enough granularity.
            successes = max(0.0, mean) * temperature_activity(temperature_c) * 10.0
            failures = max(0.0, 1.0 - mean) * (1.0 - temperature_activity(temperature_c)) * 10.0
            alpha = 1.0 + successes
            beta = 1.0 + failures
            return rng.betavariate(alpha, beta)

        scores = {a: thompson_score(a) for a in actions}
        chosen = max(scores, key=scores.get)
        return BanditAction(
            action_id=chosen,
            propensity=1.0 / len(actions),
            expected_reward=_reward(chosen),
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    # ------------------------------------------------------------------
    # Linear UCB (default)
    # ------------------------------------------------------------------
    scale = _temperature_scaled_scale(context, temperature_c)

    def ucb_score(a: str) -> float:
        """UCB = empirical mean + temperature‑scaled exploration term."""
        mean = _reward(a)
        n = _count(a)
        exploration = scale / math.sqrt(1.0 + n)
        return mean + exploration

    scores = {a: ucb_score(a) for a in actions}
    chosen = max(scores, key=scores.get)
    return BanditAction(
        action_id=chosen,
        propensity=1.0 / len(actions),
        expected_reward=_reward(chosen),
        confidence_bound=scale / math.sqrt(1.0 + _count(chosen)),
        algorithm=algorithm,
    )


def hybrid_update_policy(
    updates: List[BanditUpdate],
    temperature_c: float,
) -> None:
    """
    Update the global policy with temperature‑weighted rewards.

    Each reward is multiplied by the activity gate A(T) so that observations
    collected under sub‑optimal temperatures have proportionally less impact
    on the learned statistics.

    Parameters
    ----------
    updates: list of :class:`BanditUpdate` objects
    temperature_c: current operating temperature in Celsius
    """
    activity = temperature_activity(temperature_c)
    weighted_updates = [
        BanditUpdate(
            context_id=u.context_id,
            action_id=u.action_id,
            reward=u.reward * activity,
            propensity=u.propensity,
        )
        for u in updates
    ]
    update_policy(weighted_updates)