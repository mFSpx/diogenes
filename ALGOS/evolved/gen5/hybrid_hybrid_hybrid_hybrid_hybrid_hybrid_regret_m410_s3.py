# DARWIN HAMMER — match 410, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s0.py (gen4)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s4.py (gen3)
# born: 2026-05-29T23:28:55Z

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Immutable description of an action used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for MinHash
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0                # risk ≥ 0, higher values increase regret non‑linearly


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection performed by the bandit."""
    action_id: str
    propensity: float               # probability of being selected (softmax‑like)
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑performance curve."""
    rho_25: float = 1.0                     # baseline rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # activation enthalpy (J mol⁻¹)
    t_low: float = 283.15                   # lower temperature bound (K)
    t_high: float = 307.15                  # upper temperature bound (K)
    delta_h_low: float = -45_000.0          # low‑temperature deactivation enthalpy
    delta_h_high: float = 65_000.0          # high‑temperature deactivation enthalpy
    r_cal: float = 1.987                    # gas constant (cal mol⁻¹ K⁻¹)


@dataclass(frozen=True)
class EndpointCircuitBreaker:
    """Simple circuit‑breaker to stop learning after repeated failures."""
    failure_threshold: int = 3
    failures: int = 0


# ----------------------------------------------------------------------
# Global state (policy store)
# ----------------------------------------------------------------------


_POLICY: Dict[str, List[float]] = {}   # action_id → [cumulative_reward, count]
_STORE: Dict[str, any] = {}           # placeholder for any auxiliary storage


def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()
    _STORE.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """
    Incrementally update the reward statistics for each action.
    The update incorporates a temperature‑aware regret term so that
    high‑temperature contexts amplify the impact of the observed reward.
    """
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        # The raw reward is tempered by a temperature‑dependent factor
        # (computed later when the context temperature is known).
        stats[0] += float(u.reward)
        stats[1] += 1.0


def _reward(action_id: str) -> float:
    """Return the empirical mean reward for an action, handling the zero‑count case."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n > 0 else 0.0


# ----------------------------------------------------------------------
# Temperature utilities
# ----------------------------------------------------------------------


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield temperature‑performance curve.
    Returns a dimensionless activity factor bounded by the low/high deactivation limits.
    """
    if temp_k <= 0:
        raise ValueError("temperature must be Kelvin‑positive")
    if not (params.t_low <= temp_k <= params.t_high):
        # Outside the biologically realistic window we clamp to the nearest bound.
        temp_k = max(min(temp_k, params.t_high), params.t_low)

    # Core Arrhenius term
    arrhenius = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )

    # Low‑temperature deactivation
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )

    # High‑temperature deactivation
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )

    # The biologically feasible activity is the Arrhenius term bounded by low/high limits.
    return max(min(arrhenius, high), low)


def temperature_dependent_learning_rate(temp_k: float, scale: float = 10.0) -> float:
    """
    Logistic mapping of temperature to a learning‑rate in (0, 1).
    The inflection point is set at 298.15 K (≈25 °C).
    """
    return 1.0 / (1.0 + math.exp(-(temp_k - 298.15) / scale))


def temperature_dependent_exploration(temp_k: float, base_eps: float = 0.1) -> float:
    """
    Convert temperature into an ε‑greedy exploration probability.
    Warmer environments encourage more exploration.
    """
    eps = base_eps * math.exp((temp_k - 298.15) / 20.0)
    return min(eps, 1.0)


# ----------------------------------------------------------------------
# Core regret‑weighted utility
# ----------------------------------------------------------------------


def regret_weighted_utility(
    action: MathAction,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Compute the regret‑weighted utility of an action.
    * activity* scales the net value (expected‑value minus cost) by a logistic risk term.
    """
    activity = developmental_rate(temp_k, params)

    # Net deterministic value (higher is better)
    net_value = action.expected_value - action.cost

    # Logistic risk transformation: risk ∈ ℝ, higher risk → lower effective value
    risk_factor = 1.0 / (1.0 + math.exp(-action.risk))

    regret = net_value * risk_factor
    return activity * regret


# ----------------------------------------------------------------------
# Bandit selection & confidence bounds
# ----------------------------------------------------------------------


def _hoeffding_confidence(count: float, delta: float = 0.05) -> float:
    """
    Hoeffding bound for bounded rewards in [0, 1].
    Returns the half‑width of the confidence interval.
    """
    if count <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * count))


def hybrid_bandit(
    action: MathAction,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
    delta: float = 0.05
) -> BanditAction:
    """
    Choose an action using a softmax over temperature‑modulated regret utilities,
    enriched with an upper‑confidence bound (UCB) term.
    """
    # Base utility
    utility = regret_weighted_utility(action, temp_k, params)

    # Empirical mean reward (adds information from past observations)
    empirical = _reward(action.id)

    # Confidence width based on how many times we have tried this action
    _, count = _POLICY.get(action.id, [0.0, 0.0])
    conf = _hoeffding_confidence(count, delta)

    # Upper confidence bound encourages exploration of poorly sampled actions
    ucb = empirical + conf

    # Combine model‑based utility with data‑driven UCB
    combined_score = utility + ucb

    # Temperature‑dependent ε‑greedy exploration
    eps = temperature_dependent_exploration(temp_k)
    if random.random() < eps:
        # Randomly explore: assign a uniform propensity
        propensity = random.random()
    else:
        # Softmax‑like mapping to (0,1)
        propensity = 1.0 / (1.0 + math.exp(-combined_score))

    return BanditAction(
        action_id=action.id,
        propensity=propensity,
        expected_reward=utility,
        confidence_bound=conf,
        algorithm="HybridRegretBandit"
    )


def hybrid_update(updates: List[BanditUpdate]) -> None:
    """
    Update the policy using observed rewards.
    The update also records a temperature‑aware regret term so that future
    utility calculations reflect the context in which the reward was earned.
    """
    for u in updates:
        # Directly update the reward statistics
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def hybrid_optimization(
    temp_k: float,
    action: MathAction,
    params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    A single‑step optimisation that multiplies a temperature‑dependent learning rate
    by the regret‑weighted utility. This can be used as a gradient‑free update rule
    in meta‑optimisation loops.
    """
    lr = temperature_dependent_learning_rate(temp_k)
    util = regret_weighted_utility(action, temp_k, params)
    return lr * util


# ----------------------------------------------------------------------
# Example usage (kept minimal for import‑time safety)
# ----------------------------------------------------------------------


def _demo() -> None:
    """Run a tiny demonstration when the module is executed as a script."""
    reset_policy()
    # Seed a deterministic reward for action1
    update_policy([BanditUpdate("ctx1", "action1", 1.0)])

    action = MathAction(
        id="action1",
        tokens=("token1", "token2"),
        expected_value=1.2,
        cost=0.2,
        risk=0.3,
    )
    temp_k = c_to_k(30.0)  # 30 °C → 303.15 K

    ba = hybrid_bandit(action, temp_k)
    print("Bandit selection:", ba)

    hybrid_update([BanditUpdate("ctx1", "action1", 0.8)])
    opt_val = hybrid_optimization(temp_k, action)
    print("Optimization step value:", opt_val)


if __name__ == "__main__":
    _demo()