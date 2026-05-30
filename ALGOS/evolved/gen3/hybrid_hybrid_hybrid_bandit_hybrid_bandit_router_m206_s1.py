# DARWIN HAMMER — match 206, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3.py (gen2)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s5.py (gen1)
# born: 2026-05-29T23:27:44Z

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Bandit core – global statistics
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


# Global per‑action statistics: action_id → [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}

# Global mapping from action_id to its associated temperature (°C)
_ACTION_TEMPS: Dict[str, float] = {}

# Online linear‑regression parameters linking model‑based rate to observed reward
_BETA: float = 1.0  # slope estimate
_BETA_SUM_XX: float = 0.0  # Σ x_i²
_BETA_SUM_XY: float = 0.0  # Σ x_i y_i


def reset_policy() -> None:
    """Clear all stored reward statistics and regression state."""
    _POLICY.clear()
    _ACTION_TEMPS.clear()
    global _BETA, _BETA_SUM_XX, _BETA_SUM_XY
    _BETA = 1.0
    _BETA_SUM_XX = 0.0
    _BETA_SUM_XY = 0.0


def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def _count(a: str) -> float:
    """Number of times action *a* has been observed."""
    return _POLICY.get(a, [0.0, 0.0])[1]


def update_policy(updates: Sequence[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Schoolfield temperature model (Parent B)
# ----------------------------------------------------------------------


R_CAL = 1.987  # cal mol⁻¹ K⁻¹
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
    """
    R = params.r_cal
    T = temp_k

    act = math.exp(-params.delta_h_activation / (R * T))
    low = math.exp(params.delta_h_low / R * (1.0 / params.t_low - 1.0 / T))
    high = math.exp(params.delta_h_high / R * (1.0 / T - 1.0 / params.t_high))

    denom = 1.0 + low + high
    return params.rho_25 * act / denom


# ----------------------------------------------------------------------
# Helper utilities for the hybrid model
# ----------------------------------------------------------------------


def register_actions(actions: Sequence[str], temps_c: Sequence[float]) -> None:
    """
    Register a set of actions together with their associated temperatures (°C).
    This populates the global ``_ACTION_TEMPS`` mapping used for model‑based
    predictions and beta updates.
    """
    if len(actions) != len(temps_c):
        raise ValueError("actions and temps_c must have the same length")
    _ACTION_TEMPS.clear()
    _ACTION_TEMPS.update(dict(zip(actions, temps_c)))


def compute_model_rates(
    actions: Sequence[str],
    params: SchoolfieldParams = SchoolfieldParams(),
) -> Dict[str, float]:
    """
    Return a mapping ``action_id → model_rate`` using the registered temperatures.
    """
    rates: Dict[str, float] = {}
    for act in actions:
        t_c = _ACTION_TEMPS.get(act)
        if t_c is None:
            raise KeyError(f"Temperature for action {act!r} not registered")
        rates[act] = schoolfield_rate(c_to_k(t_c), params)
    return rates


def _update_beta(x: float, y: float) -> None:
    """
    Online update of the linear relationship ``reward ≈ β * model_rate``.
    Uses ordinary least‑squares sufficient statistics.
    """
    global _BETA, _BETA_SUM_XX, _BETA_SUM_XY
    _BETA_SUM_XX += x * x
    _BETA_SUM_XY += x * y
    if _BETA_SUM_XX > 0.0:
        _BETA = _BETA_SUM_XY / _BETA_SUM_XX


def _beta_std_error() -> float:
    """
    Rough standard‑error estimate for β based on the accumulated design matrix.
    """
    if _BETA_SUM_XX == 0.0:
        return float("inf")
    return 1.0 / math.sqrt(_BETA_SUM_XX)


# ----------------------------------------------------------------------
# Hybrid decision functions
# ----------------------------------------------------------------------


def select_action(
    context: Dict[str, float],
    actions: Sequence[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> BanditAction:
    """
    Choose an action using a bandit policy whose expected reward is a
    *scaled* Schoolfield rate. The scaling factor β is learned online.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Model‑based rates (deterministic part)
    model_rates = compute_model_rates(actions, params)

    # Context‑dependent scale for the LinUCB exploration term
    scale = math.sqrt(sum(v * v for v in context.values())) if context else 1.0

    def score(a: str) -> float:
        # Predicted reward = β * model_rate + empirical residual (optional)
        pred = _BETA * model_rates[a]
        exploration = 0.1 * scale / math.sqrt(1.0 + _count(a))
        return pred + exploration

    # Epsilon‑greedy fallback
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(list(actions))
    else:
        chosen = max(actions, key=score)

    propensity = 1.0 / len(actions)
    confidence = _beta_std_error() / math.sqrt(1.0 + _count(chosen))
    expected = _BETA * model_rates[chosen]

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected,
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def hybrid_step(
    context: Dict[str, float],
    actions: Sequence[str],
    true_reward_fn: Callable[[str, float], float],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> Tuple[BanditAction, BanditUpdate]:
    """
    Perform a single decision‑observation cycle:

    1. Select an action with ``select_action``.
    2. Query ``true_reward_fn`` for a stochastic reward (receives action_id and its
       temperature in °C).
    3. Update the global policy and the β regression using the observed reward.
    """
    # Ensure the temperature mapping is available for the current action set
    register_actions(actions, [ _ACTION_TEMPS.get(a, 0.0) for a in actions ])

    ba = select_action(
        context,
        actions,
        algorithm=algorithm,
        epsilon=epsilon,
        seed=seed,
        params=params,
    )

    # Retrieve temperature for the chosen action
    temp_c = _ACTION_TEMPS[ba.action_id]
    reward = true_reward_fn(ba.action_id, temp_c)

    # Update empirical statistics
    upd = BanditUpdate(
        context_id=str(context),
        action_id=ba.action_id,
        reward=reward,
        propensity=ba.propensity,
    )
    update_policy([upd])

    # Update the β regression using the model rate for the chosen action
    model_rate = schoolfield_rate(c_to_k(temp_c), params)
    _update_beta(model_rate, reward)

    return ba, upd


# ----------------------------------------------------------------------
# Simple stochastic reward generator for testing
# ----------------------------------------------------------------------


def make_noisy_reward_fn(
    noise_std: float = 0.05,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> Callable[[str, float], float]:
    """
    Return a callable that maps ``(action_id, temperature_C)`` to a noisy
    observation of the underlying Schoolfield rate.
    """
    rng = np.random.default_rng()

    def fn(_: str, temp_c: float) -> float:
        true_rate = schoolfield_rate(c_to_k(temp_c), params)
        return float(rng.normal(true_rate, noise_std * max(1.0, true_rate)))

    return fn