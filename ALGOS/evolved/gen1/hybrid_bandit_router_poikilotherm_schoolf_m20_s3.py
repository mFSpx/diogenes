# DARWIN HAMMER — match 20, survivor 3
# gen: 1
# parent_a: bandit_router.py (gen0)
# parent_b: poikilotherm_schoolfield.py (gen0)
# born: 2026-05-29T23:23:03Z

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
            failures = 10.0 - successes
            alpha = 1.0 + successes
            beta = 1.0 + failures
            return rng.betavariate(alpha, beta)

        scores = {a: thompson_score(a) for a in actions}
        chosen = max(scores, key=scores.get)
        return BanditAction(
            action_id=chosen,
            propensity=1.0,
            expected_reward=_reward(chosen),
            confidence_bound=0.0,
            algorithm=algorithm,
        )

    # ---- LinUCB ---------------------------------------------------------
    if algorithm != "linucb":
        raise ValueError(f"Unknown algorithm: {algorithm}")

    temperature_scaled_scale = _temperature_scaled_scale(context, temperature_c)

    # Compute UCB for each action
    ucbs = []
    for a in actions:
        mean = _reward(a)
        n_a = _POLICY.get(a, [0.0, 0.0])[1]
        if n_a == 0:
            ucb = float('inf')
        else:
            ucb = mean + temperature_scaled_scale * math.sqrt(2 * math.log(1.0 / 0.1) / n_a)
        ucbs.append((a, ucb))

    # Select action with highest UCB
    chosen, _ = max(ucbs, key=lambda x: x[1])
    return BanditAction(
        action_id=chosen,
        propensity=1.0,
        expected_reward=_reward(chosen),
        confidence_bound=temperature_scaled_scale * math.sqrt(2 * math.log(1.0 / 0.1) / _POLICY.get(chosen, [0.0, 0.0])[1]) if _POLICY.get(chosen, [0.0, 0.0])[1] > 0 else 0.0,
        algorithm=algorithm,
    )

def hybrid_update_policy(updates: List[BanditUpdate]) -> None:
    """Update policy with a batch of observations."""
    update_policy(updates)