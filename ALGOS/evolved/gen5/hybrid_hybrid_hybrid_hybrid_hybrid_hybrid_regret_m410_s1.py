# DARWIN HAMMER — match 410, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s0.py (gen4)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s4.py (gen3)
# born: 2026-05-29T23:28:55Z

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for MinHash
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


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
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy (not used directly in the demo)."""
    context_id: str
    action_id: str
    reward: float


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


@dataclass(frozen=True)
class EndpointCircuitBreaker:
    failure_threshold: int = 3
    failures: int = 0


_POLICY = {}
_STORE = {}


def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()


def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, {"total": 0.0, "n": 0})
        s["total"] += float(u.reward)
        s["n"] += 1.0


def _reward(a: str) -> float:
    s = _POLICY.get(a, {"total": 0.0, "n": 0})
    return s["total"] / s["n"] if s["n"] else 0.0


def c_to_k(celsius: float) -> float:
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    return min(max(numerator, low), high)


def regret_weighted_utility(action: MathAction, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Regret-weighted utility of an action modulated by the temperature-dependent activity curve."""
    activity = developmental_rate(temp_k, params)
    regret = (action.expected_value - action.cost) / (1 + math.exp(-action.risk))
    return activity * regret


def hybrid_bandit(action: MathAction, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> BanditAction:
    """Hybrid bandit action selection using regret-weighted utility modulated by temperature-dependent activity curve."""
    utility = regret_weighted_utility(action, temp_k, params)
    propensity = 1.0 / (1.0 + np.exp(-utility))
    return BanditAction(action.id, propensity, utility, 0.0, "HybridRegretBandit")


def hybrid_update(updates: list[BanditUpdate]) -> None:
    """Hybrid policy update using regret-weighted utility modulated by temperature-dependent activity curve."""
    for u in updates:
        _reward(u.action_id)


def temperature_dependent_learning_rate(temp_k: float) -> float:
    """Temperature-dependent learning rate."""
    return 1.0 / (1.0 + math.exp(-(temp_k - 298.15) / 10.0))


def hybrid_optimization(temp_k: float, action: MathAction) -> float:
    """Hybrid optimization using temperature-dependent learning rate and regret-weighted utility."""
    learning_rate = temperature_dependent_learning_rate(temp_k)
    utility = regret_weighted_utility(action, temp_k)
    return learning_rate * utility


def kl_divergence(p: dict, q: dict) -> float:
    return sum(p[x] * math.log(p[x] / q[x]) for x in p if x in q)


def hybrid_bandit_kl(action1: MathAction, action2: MathAction, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    p = {"action1": regret_weighted_utility(action1, temp_k, params), 
         "action2": regret_weighted_utility(action2, temp_k, params)}
    p = {k: v / sum(p.values()) for k, v in p.items()}
    q = {"action1": 0.5, "action2": 0.5}
    return kl_divergence(p, q)


if __name__ == "__main__":
    reset_policy()
    update_policy([BanditUpdate("context1", "action1", 1.0)])
    action1 = MathAction("action1", ("token1",), 1.0)
    action2 = MathAction("action2", ("token2",), 1.2)
    temp_k = c_to_k(300.0)
    print(hybrid_bandit(action1, temp_k))
    print(hybrid_update([BanditUpdate("context1", "action1", 1.0)]))
    print(hybrid_optimization(temp_k, action1))
    print(hybrid_bandit_kl(action1, action2, temp_k))